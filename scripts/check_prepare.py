#!/usr/bin/env python3
"""Offline integration check: prepare all branches while forbidding old sources."""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    template = json.loads((ROOT/"deploy/sam/config.template.json").read_text())
    checkpoint = template["checkpoint_sha256"]
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"phase":"ready","warmup_passed":True,"model":{
                "checkpoint_hash_verified":True,"checkpoint_sha256":checkpoint,
                "checkpoint":{"epoch":7},
                "checkpoint_path":str(ROOT/"weights/sam3_best_epoch7/best.pt")}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1",0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="png2pptx-handoff-prepare-") as directory:
            tmp=Path(directory)
            data=tmp/"input"
            (data/"tasks/smoke").mkdir(parents=True)
            (data/"taskset.txt").write_text("smoke\n")
            Image.new("RGB",(64,48),"white").save(data/"tasks/smoke/reference.png")
            key=tmp/"key"
            key.write_text("offline-check-placeholder\n")
            key.chmod(0o600)
            run=tmp/"run"
            bootstrap=tmp/"bootstrap.py"
            bootstrap.write_text(
                "import os,runpy,sys\nfrom pathlib import Path\n"+
                "root=Path("+repr(str(ROOT))+")\n"+
                "forbidden=[str(root.parent),'/mnt/shared-storage-user/intern7shared/yinjinhui/scifig_png2ppt/ref/png2pptx_geometry']\n"+
                "def audit(event,args):\n"+
                " if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):\n"+
                "  p=os.path.abspath(os.fsdecode(args[0]))\n"+
                "  if not (p==str(root) or p.startswith(str(root)+'/')) and any(p==f or p.startswith(f+'/') for f in forbidden):\n"+
                "   raise RuntimeError('External original source accessed: '+p)\n"+
                "sys.addaudithook(audit)\nsys.path.insert(0,str(root))\n"+
                "sys.argv=[str(root/'scripts/prepare_forward.py'),*sys.argv[1:]]\n"+
                # run_ppt must execute in the audited process, not a subprocess.
                "sys.argv=[str(root/'run_ppt.py'),'--prepare','--api',*sys.argv[1:]]\n"+
                "runpy.run_path(str(root/'run_ppt.py'),run_name='__main__')\n")
            result=subprocess.run([sys.executable,"-B",str(bootstrap),
                "--input-root",str(data),"--run-dir",str(run),
                "--sam-url","http://127.0.0.1:"+str(server.server_port),
                "--api-base-url","http://127.0.0.1:9/v1","--api-key-file",str(key)],
                cwd=ROOT, text=True, capture_output=True,
                env={**os.environ,"PNG2PPTX_CPU_ENV":str(tmp/"cpu-env")})
            if result.returncode:
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise SystemExit(result.returncode)
            settings=json.loads((run/"settings.json").read_text())
            assert settings["arrow_driver"]=="frozen_geometry_first"
            assert settings["external_sam_epoch"]==7
            assert not settings.get("external_results_reused",False)
            arrow_root=Path(settings["fresh_arrow_root"])
            arrow_config=json.loads((arrow_root/"config.json").read_text())
            assert not arrow_config.get("localization_reuse_from")
            assert not arrow_config.get("comparison_run")
            for relative in ("config.json","localization/config.json","sam/missing/config.json"):
                active=json.loads((arrow_root/relative).read_text())
                def check_paths(value):
                    if isinstance(value,dict):
                        for item in value.values():check_paths(item)
                    elif isinstance(value,list):
                        for item in value:check_paths(item)
                    elif isinstance(value,str) and value.startswith(("/mnt/","/data/","/opt/")):
                        assert value.startswith(str(ROOT)+"/") or value.startswith(str(tmp)+"/"), value
                check_paths(active)
            # A deployment outside the original cluster has neither the training
            # dataset directory nor a CPU environment mounted at /opt/shape-cpu.
            categories=[c["name"] for c in json.loads((ROOT/"deploy/sam/categories.json").read_text())["categories"]]
            for name in ("settings.json","shape/settings.json"):
                active=json.loads((run/name).read_text())
                dataset_root=Path(active["dataset_root"])
                assert dataset_root.is_relative_to(run.resolve()), dataset_root
                manifest=json.loads((dataset_root/"dataset_manifest.json").read_text())
                assert [c["name"] for c in manifest["categories"]]==categories
                assert active["text_ocr_python"]==str(tmp/"cpu-env/bin/python"), active["text_ocr_python"]
            assert (run/"backends/text/vendor/png2pptx").is_dir()
            assert not (run/"backends/text/.deps").is_symlink()
            print(json.dumps({"passed":True,"scope":"fresh PNG preparation of shape, text, arrow and composition",
                "original_source_access_blocked":True,"active_arrow_config_has_no_old_run_paths":True,
                "sam_health":"local stub only",
                "real_model_inference":False,"arrow_driver":settings["arrow_driver"]},indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
