"""Download the NSL-KDD files used in this project into data/.

Usage:
    python download_data.py

Why a script instead of curl commands: it works the same on Windows,
macOS and Linux, creates the data/ folder, and checks each file's
SHA-256 fingerprint so you know you have exactly the files my results
were produced from.

Source: the official NSL-KDD page (https://www.unb.ca/cic/datasets/nsl.html)
states the dataset is no longer available there. Its licence allows
mirroring with citation, so the files come from a public GitHub mirror.
Please cite Tavallaee et al. (2009) if you use the data.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MIRROR = "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/"

# File name -> SHA-256 of the copy used for the published results.
FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}


def sha256_of(path: Path) -> str:
    """SHA-256 fingerprint of a file, read in chunks to keep memory use low."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, expected_sha256: str) -> bool:
    """Download one file (unless a verified copy is already present). Return True if it verifies."""
    target = DATA_DIR / name
    if target.exists() and sha256_of(target) == expected_sha256:
        print(f"  {name}: already present and verified")
        return True

    print(f"  {name}: downloading ...")
    urllib.request.urlretrieve(MIRROR + name, target)
    if sha256_of(target) != expected_sha256:
        print(f"  {name}: WARNING - checksum does not match the file used for the results")
        return False
    print(f"  {name}: downloaded and verified ({target.stat().st_size:,} bytes)")
    return True


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Saving NSL-KDD files to {DATA_DIR}")
    all_ok = all([download(name, sha) for name, sha in FILES.items()])
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
