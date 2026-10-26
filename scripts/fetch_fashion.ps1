param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$fashionFiles = [ordered]@{
    'train-images-idx3-ubyte.gz' = '8d4fb7e6c68d591d4c3dfef9ec88bf0d'
    'train-labels-idx1-ubyte.gz' = '25c81989df183df01b3e8a0aad5dffbe'
    't10k-images-idx3-ubyte.gz' = 'bef4ecab320f06d8554ea6380940ec79'
    't10k-labels-idx1-ubyte.gz' = 'bb300cfdad3c16e7a12a480ee83cd310'
}
$fashionCommit = (Invoke-RestMethod -Uri 'https://api.github.com/repos/zalandoresearch/fashion-mnist/commits/master' -TimeoutSec 45).sha
if ($fashionCommit -notmatch '^[0-9a-f]{40}$') { throw 'Invalid repository revision' }
$null = New-Item -ItemType Directory -Path $OutputDirectory -Force
$fashionRecords = [ordered]@{}
foreach ($fashionName in $fashionFiles.Keys) {
    $fashionTarget = Join-Path $OutputDirectory $fashionName
    $fashionUrl = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/$fashionCommit/data/fashion/$fashionName"
    if (-not (Test-Path -LiteralPath $fashionTarget)) {
        Write-Output "Downloading $fashionName"
        Invoke-WebRequest -Uri $fashionUrl -OutFile $fashionTarget -TimeoutSec 45
    }
    $fashionMd5 = (Get-FileHash -LiteralPath $fashionTarget -Algorithm MD5).Hash.ToLowerInvariant()
    if ($fashionMd5 -ne $fashionFiles[$fashionName]) { throw "Integrity mismatch: $fashionName" }
    $fashionRecords[$fashionName] = [ordered]@{
        url = $fashionUrl
        bytes = (Get-Item -LiteralPath $fashionTarget).Length
        md5 = $fashionMd5
        sha256 = (Get-FileHash -LiteralPath $fashionTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    Write-Output "Verified $fashionName"
}
$fashionLicenseUrl = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/$fashionCommit/LICENSE"
Invoke-WebRequest -Uri $fashionLicenseUrl -OutFile (Join-Path $OutputDirectory 'LICENSE.fashion-mnist') -TimeoutSec 45
$fashionAcquisition = [ordered]@{
    acquired_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    source_repository = 'https://github.com/zalandoresearch/fashion-mnist'
    source_revision = $fashionCommit
    method = 'Windows Invoke-WebRequest HTTPS with certificate validation; published MD5 and recorded SHA256'
    files = $fashionRecords
    license = [ordered]@{
        url = $fashionLicenseUrl
        sha256 = (Get-FileHash -LiteralPath (Join-Path $OutputDirectory 'LICENSE.fashion-mnist') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$fashionAcquisition | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'acquisition.json') -Encoding utf8
Write-Output 'PASS: four official files and license retrieved and verified; no predictions.'
