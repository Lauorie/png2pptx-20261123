"""Compatibility CLI for an already decided image plan; never re-extract assets."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.backend/'src'),str(a.backend/'vendor')]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from full_recovery.image_assets import commit
    from shape_integration.common import save,sha
    job=json.loads(a.job.read_text());folder=Path(job['folder'])
    if 'assets' not in job:
        raise ValueError('Image stage requires the unified asset plan; a raw registry is not an insertion queue')
    assets=job['assets']
    for asset in assets:
        if asset['source_sha256']!=job['item']['source_sha256'] or sha(Path(asset['path']))!=asset['sha256']:
            raise ValueError('Texture fallback source/asset identity changed')
    placed=commit(Path(job['base']),Path(job['output']),assets,job['item']['source_size'])
    save(folder/'validation.json',{'source_sha256':job['item']['source_sha256'],'assets':assets,
          'placement':placed,'output_sha256':sha(Path(job['output']))})


if __name__=='__main__':main()
