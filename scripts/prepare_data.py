                                                                                      

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.data import audit_original_data, load_training_pool, make_split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    pool = load_training_pool(args.data_dir)
    manifest = make_split_manifest(pool)
    audit = audit_original_data(args.data_dir, manifest)
    source_paths = ["src/si_audit/data.py", "scripts/prepare_data.py", "EXPERIMENT_PROTOCOL.md"]
    artifact = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "acquisition": json.loads((args.data_dir / "acquisition.json").read_text(encoding="utf-8")),
        "audit": audit, "split_manifest": manifest,
    }
    payload = json.dumps(artifact, indent=2) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(json.dumps({"saved": str(destination.resolve()), "audit": audit}, indent=2))
    if not audit["audit_gate_passed"]:
        raise SystemExit("Audit gate failed: amend the protocol before fitting models")


if __name__ == "__main__":
    main()
