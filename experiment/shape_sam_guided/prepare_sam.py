"""Cache segmentation once, then release its GPUs for the longer GLM phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from core import read,save
from models import Models
from run import sam_index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);models=Models(config)
    items=read(Path(config['baseline_root'])/'inputs/manifest.json')['images'];rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(sam_index,i,config,models,args.run_dir/'sam_cache'/i['id']):i for i in items}
        for future in as_completed(pending):
            item=pending[future];index=future.result()
            rows.append({'id':item['id'],'groups':index['group_count'],'elapsed_seconds':index['elapsed_seconds'],
                         'reused':bool(index.get('cache_source'))})
            save(args.run_dir/'sam_precompute.json',{'completed':len(rows),'expected':len(items),'results':rows})
            print('SAM_CACHED',item['id'],index['group_count'],flush=True)


if __name__=='__main__':main()
