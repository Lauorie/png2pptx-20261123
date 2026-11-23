"""Read-only partial review while the frozen experiment is running."""
import argparse,copy,time,os,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import audit
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import viz
from transport.common import read_json as read,write_json as write


def main(run):
 run=Path(run).resolve();previous={};ids=read(run/'manifest.json')['ids'];pid=read(run/'process.json')['pid']
 def one(cid):
  source=run/'cases'/cid;paths=sorted(list(source.glob('groups/*/completed.json'))+list(source.glob('missing/*/completed.json')))
  marker=(len(paths),(source/'completed.json').exists())
  if previous.get(cid)==marker:return
  rows=[]
  for p in paths:rows+=read(p).get('rows',[])
  if not paths:return
  out=run/'analysis/live'/cid;out.mkdir(parents=True,exist_ok=True)
  labels={key:f'A{i}' for i,key in enumerate(sorted({r['parent_id'] for r in rows}),1)}
  for r in rows:r['label']=labels[r['parent_id']]
  viz.render(source/'clean.png',rows,out)
  write(out/'snapshot.json',dict(completed_groups=len(paths),case_finished=marker[1],rows=rows))
  previous[cid]=marker
 while True:
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,ids))
  progress=read(run/'progress.json');lines=['# 实时预览（未结束的 case 仅显示已完成组）','',
   f'整图阶段已结束 {progress["completed"]}/{len(ids)}。空白部分可能仍在运行，不能据此判定漏检。最终结果请看 [RESULTS.md](RESULTS.md)。','',
   '| case | 已完成组 | 原图 | 彩色实例 | 样式 | 叠图 |','|---|---:|---|---|---|---|']
  for cid in ids:
   if cid not in previous:continue
   prefix=f'analysis/live/{cid}'
   lines.append(f'| {cid} | {previous[cid][0]} | ![](cases/{cid}/clean.png) | ![]({prefix}/instances.png) | ![]({prefix}/styled.png) | ![]({prefix}/overlap.png) |')
  (run/'LIVE.md.tmp').write_text('\n'.join(lines)+'\n');(run/'LIVE.md.tmp').replace(run/'LIVE.md')
  audit.audit(run)
  if (run/'finished.json').exists():break
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(30)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');main(p.parse_args().run)
