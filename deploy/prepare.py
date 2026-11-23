#!/usr/bin/env python3
"""Prepare a fresh deployment directory from the bundled source and weights."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("sam", "sensenova"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sensenova-python", type=Path, default=ROOT/".runtime/sensenova/bin/python")
    parser.add_argument("--base-environment", type=Path, default=Path("/mnt/shared-storage-user/yinjinhui/H-conda/anaconda3/envs/openrlhf"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists():
        parser.error("Use a new deployment directory: " + str(run))
    source = ROOT/"deploy"/args.service
    if args.service == "sam":
        config = json.loads((source/"config.template.json").read_text())
        repo = source/"vendor/sam3"
        config.update(training_run=str(source), sam_repository=str(repo),
                      dataset_manifest=str(source/"categories.json"),
                      checkpoint_path=str(ROOT/"weights/sam3_best_epoch7/best.pt"))
        for name, asset in config["runtime_assets"].items():
            asset["path"] = str(repo/name)
        code_name = "code"
    else:
        config = json.loads((source/"config.json").read_text())
        for key in ("baseline_run", "sam_source", "runtime", "source_run", "created_at", "job"):
            config.pop(key, None)
        config.update(runtime_python=str(args.sensenova_python.absolute()),
                      base_environment=str(args.base_environment.resolve()))
        code_name = "code_snapshot"
    config.update(service=args.service, bundle_root=str(ROOT))
    checkpoint = ROOT/("weights/sam3_best_epoch7/best.pt" if args.service == "sam"
                      else "weights/sensenova/SenseNova-U1.5-8B-MoT/config.json")
    if not checkpoint.is_file():
        parser.error("Missing bundled model: " + str(checkpoint))
    run.mkdir(parents=True)
    shutil.copytree(source/code_name, run/code_name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (run/"config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    (run/"jobs").mkdir()
    if args.service == "sensenova":
        (run/"vendor").symlink_to(source/"vendor", target_is_directory=True)
        (run/"evidence").mkdir()
        for index in range(2):
            (run/"power"/str(index)).mkdir(parents=True)
    print(json.dumps({"service":args.service, "run_dir":str(run),
                      "next":"bash deploy/submit.sh "+str(run)}, indent=2))


if __name__ == "__main__":
    main()
