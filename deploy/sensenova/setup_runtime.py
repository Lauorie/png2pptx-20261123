#!/usr/bin/env python3
"""Rebuild the tested Python 3.12 overlay using a mounted base environment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Entries of the original base environment that the service never imports. A
# base rebuilt from base_environment_versions.json does not contain them.
NONFUNCTIONAL = {"README.txt", "hopper", "platformdirs-4.9.2.dist-info", "pip-26.0.1.dist-info",
                 "setuptools-80.10.2-py3.12.egg-info", "__editable___openrlhf_0_9_5_finder.py",
                 "openrlhf-0.9.5.dist-info", "pygments-2.19.2.dist-info"}


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
    unexpected = [name for name in missing if name not in NONFUNCTIONAL]
    if unexpected:
        parser.error("Base environment differs from the frozen runtime; missing packages: "+", ".join(unexpected))
    # The overlay replaces some base distributions with other versions. The
    # base copy's metadata must not be linked beside them: version checks
    # (transformers on huggingface-hub) would read the base version.
    def distribution(name):
        match = re.match(r"(.+?)-\d[^-]*\.(dist-info|egg-info)$", name)
        return re.sub(r"[-_.]+", "_", match[1]).lower() if match else None
    overlaid = {distribution(path.name) for path in (HERE/"runtime_overlay").iterdir()} - {None}
    shadowed = sorted(name for name in spec["linked_packages"]
                      if distribution(name) in overlaid and not (HERE/"runtime_overlay"/name).exists())
    subprocess.run([str(base_python), "-m", "venv", "--without-pip", str(output)], check=True)
    site = output/"lib/python3.12/site-packages"
    for name in spec["linked_packages"]:
        if name == "site-packages" or name in missing or name in shadowed or (HERE/"runtime_overlay"/name).exists():
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
    # LightLLM starts its HTTP front end through the "hypercorn" command. The
    # package comes from the overlay, which carries no console scripts.
    launcher = output/"bin/hypercorn"
    launcher.write_text("#!"+str(output/"bin/python")+"\nimport sys\nfrom hypercorn.__main__ import main\n"
                        "if __name__ == \"__main__\":\n    sys.exit(main())\n")
    launcher.chmod(0o755)
    (output/"handoff_runtime.json").write_text(json.dumps({
        "base_python":str(base_python), "base_environment":info["prefix"],
        "overlay":str(HERE/"runtime_overlay"), "skipped_nonfunctional":sorted(missing),
        "skipped_shadowed_metadata":shadowed,
        "base_environment_mount_required":True}, indent=2)+"\n")
    print(output/"bin/python")


if __name__ == "__main__":
    main()
