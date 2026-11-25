"""Run one fresh shared SAM service without historical prediction artifacts."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    try: Path('/proc/self/oom_score_adj').write_text('0')
    except OSError: pass  # container forbids lowering the OOM score
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    root=parser.parse_args().run_dir.resolve()
    for name in list(os.environ):
        if name.lower() in ('http_proxy','https_proxy','all_proxy'):
            os.environ.pop(name)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,str(root/'code'))
    sys.argv=[str(root/'code/server.py'),'--config',str(root/'config.json'),'--run-dir',str(root/'runtime')]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
