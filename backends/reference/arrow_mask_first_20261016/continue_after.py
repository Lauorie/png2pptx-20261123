"""One finite cached continuation after the current 40-case run ends."""
import os,subprocess,sys,time
from pathlib import Path
from transport.common import read_json as read,write_json as write

root=Path(__file__).resolve().parent
source=root/'runs/all40_mask_geometry_v2';target=root/'runs/all40_mask_geometry_v3'
source_pid=read(source/'process.json')['pid'];state=root/'continuation.json'
def publish(stage):
 write(state,dict(pid=os.getpid(),source=str(source),source_pid=source_pid,target=str(target),stage=stage))
 current=target if stage in ('running','finished') else source
 rel=current.relative_to(root)
 (root/'RESULTS.md').write_text(f'# 箭头重建实验\n\n当前：`{rel}`。\n\n[实时预览]({rel}/LIVE.md) · [整图结果]({rel}/RESULTS.md) · [运行统计]({rel}/analysis/README.md)\n\n[首轮观察与问题](analysis/first_review/README.md)。\n\n状态：{stage}。`waiting` 表示当前全量继续运行，结束后只通过缓存续修格式问题；不会把模型坐标或身份改成规则判断。\n')
publish('waiting')
while not (source/'finished.json').exists():
 if not Path(f'/proc/{source_pid}').exists():publish('source_stopped');sys.exit(1)
 time.sleep(30)
log=(target/'continuation.log').open('a')
proc=subprocess.Popen([sys.executable,'-u',str(target/'code/run.py'),'--run',str(target),'--config',str(target/'config.json'),'--frozen'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
while not (target/'process.json').exists() and proc.poll() is None:time.sleep(1)
publish('running')
preview=subprocess.Popen([sys.executable,'-u',str(root/'live_review.py'),str(target)],cwd=root,stdout=(target/'preview.log').open('a'),stderr=subprocess.STDOUT)
status=proc.wait();publish('finished' if status==0 else 'failed');sys.exit(status)
