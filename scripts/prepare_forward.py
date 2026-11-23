#!/usr/bin/env python3
"""Prepare a hosted-model forward run using only this handoff's backends."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, action="append", required=True)
    parser.add_argument("--api-model", default="glm-5.3-flash")
    parser.add_argument("--image-edit-url")
    parser.add_argument("--ids", nargs="+")
    args, extra = parser.parse_known_args()
    command = [sys.executable, "-B", str(ROOT/"run_ppt.py"), "--prepare", "--api",
               "--input-root", str(args.input_root.resolve()), "--run-dir", str(args.run_dir.resolve()),
               "--sam-url", args.sam_url, "--api-base-url", args.api_base_url,
               "--api-model", args.api_model,
               "--arrow-native-renderer", str(ROOT/".runtime/arrow_renderer/soffice")]
    for key in args.api_key_file:
        command.extend(["--api-key-file", str(key.resolve())])
    if args.image_edit_url:
        command.extend(["--image-edit-url", args.image_edit_url])
    if args.ids:
        command.extend(["--ids", *args.ids])
    subprocess.run([*command, *extra], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
