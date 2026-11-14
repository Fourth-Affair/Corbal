"""Extract the top-level questions.jsonl from the challenge dataset zip: a local file, or a URL read with HTTP range
requests (no full download).

  python scripts/fetch_questions.py <zip-path-or-url> <out.jsonl>
"""

import sys
import zipfile
from pathlib import Path


def main() -> None:
    src, out = sys.argv[1], Path(sys.argv[2])
    if src.startswith(("http://", "https://")):
        from remotezip import RemoteZip

        z = RemoteZip(src)
    else:
        z = zipfile.ZipFile(src)
    with z:
        names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "questions.jsonl"]
        if not names:
            raise SystemExit("no questions.jsonl in the zip")
        name = min(names, key=lambda n: n.count("/"))  # the dataset's own, not a nested copy
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(name))
    print(f"{name} -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
