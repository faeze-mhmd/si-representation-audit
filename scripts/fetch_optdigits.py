                                                                                    

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
PAGE = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"
LIMIT = 2_000_000
FILES = ("optdigits.tra", "optdigits.tes", "optdigits.names")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded official archive without a network request")
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    if args.archive is not None:
        if args.archive.stat().st_size > LIMIT:
            raise ValueError("Local archive exceeds the 2 MB acquisition cap")
        payload = args.archive.read_bytes()
        final_url = None
    else:
        with urllib.request.urlopen(URL, timeout=45) as response:
            final_url = response.geturl()
            payload = response.read(LIMIT + 1)
    if len(payload) > LIMIT:
        raise ValueError("UCI archive exceeds the 2 MB acquisition cap")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        content = {}
        for name in FILES:
            info = archive.getinfo(name)
            if info.file_size > LIMIT:
                raise ValueError("Unexpectedly large member: " + name)
            content[name] = archive.read(name)
                                                                            
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        path = destination / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("Existing raw file differs; preserve and inspect: " + str(path))
    for name, data in content.items():
        (destination / name).write_bytes(data)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_url": URL,
        "resolved_url": final_url,
        "archive_input": str(args.archive.resolve()) if args.archive is not None else None,
        "acquisition_method": "local archive; see acquisition command log for download provenance" if args.archive is not None else "urllib HTTPS with certificate validation",
        "dataset_page": PAGE,
        "citation": "Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI. DOI:10.24432/C50P49",
        "license": "CC BY 4.0, as stated by the UCI dataset page",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in content.items()},
    }
    (destination / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data_dir": str(destination), **manifest}, indent=2))


if __name__ == "__main__":
    main()
