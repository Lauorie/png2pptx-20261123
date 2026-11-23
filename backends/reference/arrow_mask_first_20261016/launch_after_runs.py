"""Operational queue: preserve request capacity while prior trials drain."""
import argparse,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);a=p.parse_args()
    queue=Path(a.queue).resolve();config=json.loads(queue.read_text());out=queue.parent
    def write(state,**kw):
        tmp=out/'queue_status.tmp';tmp.write_text(json.dumps(dict(state=state,queue_pid=os.getpid(),time=time.time(),**kw),indent=2));tmp.replace(out/'queue_status.json')
    while True:
        waiting=[p for p in config['wait_for'] if not Path(p).exists()]
        if not waiting:break
        write('waiting_for_trials',remaining=waiting);time.sleep(5)
    write('launching',command=config['command'])
    with (out/'run.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(config['command'],cwd=config['cwd'],stdout=log,stderr=subprocess.STDOUT)
        write('running',pid=child.pid);code=child.wait()
    write('process_exited',pid=child.pid,exit_code=code,completed_file=(out/'finished.json').exists())


if __name__=='__main__':main()
