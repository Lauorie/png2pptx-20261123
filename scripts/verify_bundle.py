#!/usr/bin/env python3
"""Verify copied content; --weights adds a full read of the 42 GiB weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",action="store_true")
    args=parser.parse_args()
    manifest=json.loads((ROOT/"provenance/bundle_manifest.json").read_text())
    errors=[]
    checked=0
    for row in manifest["files"]:
        path=ROOT/row["path"]
        if not path.is_file():
            errors.append({"path":row["path"],"error":"missing"})
            continue
        if path.stat().st_size!=row["bytes"]:
            errors.append({"path":row["path"],"error":"size mismatch"})
        elif args.weights or not row["path"].startswith("weights/"):
            if sha(path)!=row["sha256"]:
                errors.append({"path":row["path"],"error":"SHA256 mismatch"})
            checked+=1
    for row in manifest.get("symlinks",[]):
        path=ROOT/row["path"]
        if not path.is_symlink() or str(path.readlink())!=row["target"] or not path.exists():
            errors.append({"path":row["path"],"error":"symlink mismatch or broken target"})
    print(json.dumps({"passed":not errors,"hashed_files":checked,
        "weight_hashes_checked":args.weights,"errors":errors},indent=2))
    raise SystemExit(bool(errors))


if __name__=="__main__":
    main()
