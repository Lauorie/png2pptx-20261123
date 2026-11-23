#!/usr/bin/env python3
"""Rebuild the tested Python 3.12 overlay using a mounted base environment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/".runtime/sensenova")
    args = parser.parse_args()
    base_python = args.base_python.absolute()
    info = json.loads(subprocess.check_output([str(base_python), "-c",
        "import json,sys,sysconfig; print(json.dumps(dict(version=list(sys.version_info[:2]),site=sysconfig.get_paths()['purelib'],prefix=sys.prefix)))"], text=True))
    if info["version"] != [3,12]:
        parser.error("The frozen SenseNova overlay requires CPython 3.12")
    output = args.output.absolute()
    if output.exists():
        parser.error("Use a fresh output directory: "+str(output))
    spec = json.loads((HERE/"runtime_overlay_manifest.json").read_text())
    base_site = Path(info["site"])
    missing = [name for name in spec["linked_packages"]
               if not (HERE/"runtime_overlay"/name).exists()
               and name != "site-packages" and not (base_site/name).exists()]
    if missing:
        parser.error("Base environment differs from the frozen runtime; missing packages: "+", ".join(missing))
    subprocess.run([str(base_python), "-m", "venv", "--without-pip", str(output)], check=True)
    site = output/"lib/python3.12/site-packages"
    for name in spec["linked_packages"]:
        if name == "site-packages" or (HERE/"runtime_overlay"/name).exists():
            continue
        (site/name).symlink_to(base_site/name, target_is_directory=(base_site/name).is_dir())
    shutil.copytree(HERE/"runtime_overlay", site, dirs_exist_ok=True, symlinks=True)
    for row in spec.get("relocated_links", []):
        target = row["target"].replace("{base}", info["prefix"]).replace("{runtime}", str(output))
        link = site/row["path"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target)
    (output/"handoff_runtime.json").write_text(json.dumps({
        "base_python":str(base_python), "base_environment":info["prefix"],
        "overlay":str(HERE/"runtime_overlay"),
        "base_environment_mount_required":True}, indent=2)+"\n")
    print(output/"bin/python")


if __name__ == "__main__":
    main()
