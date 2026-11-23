#!/usr/bin/env python3
"""Register a directory of original PNGs without copying their image data."""
import argparse
import json
from pathlib import Path
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    files=sorted(args.images.resolve().glob("*.png"))
    if not files:
        parser.error("No .png files in the selected directory")
    output=args.output.resolve()
    if output.exists():
        parser.error("Use a new catalog directory")
    output.mkdir(parents=True)
    (output/"images").symlink_to(args.images.resolve(), target_is_directory=True)
    (output/"catalog.json").write_text(json.dumps({"items":[
        {"case_key":p.stem,"image":"images/"+p.name} for p in files]}, ensure_ascii=False, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
