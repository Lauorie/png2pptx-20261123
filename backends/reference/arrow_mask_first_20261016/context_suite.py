"""Run matched ablations, then complete-case integration without stacked limits."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
from transport.common import read_json as read,write_json as write


def main(run,wait_for):
    run=Path(run).resolve();wait_for=Path(wait_for).resolve();started=time.time()
    def state(phase,**kw):write(run/'suite_progress.json',dict(phase=phase,pid=os.getpid(),started=started,updated=time.time(),**kw))
    state('waiting_for_running_control',control=str(wait_for))
    while not (wait_for/'finished.json').exists():
        proc=read(wait_for/'process.json')
        if not Path('/proc',str(proc['pid'])).exists():
            state('control_stopped_before_completion',control=str(wait_for));break
        state('waiting_for_running_control',control=str(wait_for),control_progress=read(wait_for/'progress.json'))
        time.sleep(20)
    def launch(name,args):
        state(name)
        with (run/(name+'.log')).open('a') as log:
            completed=subprocess.run([sys.executable,'-u',*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:raise RuntimeError(f'{name} failed with code {completed.returncode}; see {name}.log')
    launch('ablation',[run/'code/context_ablation.py','--run',run,'--prepared',run/'prepared'])
    baseline=wait_for if (wait_for/'finished.json').exists() else run.parent/'all40_corner_scope_20261017_v4'
    state('complete_case_integration')
    review_log=(run/'comparison.log').open('a')
    compare=subprocess.Popen([sys.executable,'-u',str(run/'code/compare_full_run.py'),'--run',str(run),'--baseline',str(baseline),'--watch'],stdout=review_log,stderr=subprocess.STDOUT)
    try:
        launch('complete_cases',[run/'code/run.py','--run',run,'--config',run/'config.json','--frozen','--cases',*read(run/'scope.json')['cases']])
    finally:
        # Refresh after completion even if the early watcher saw no process yet.
        subprocess.run([sys.executable,str(run/'code/compare_full_run.py'),'--run',str(run),'--baseline',str(baseline)],check=False,stdout=review_log,stderr=subprocess.STDOUT)
        review_log.close()
    state('finished',results=str(run/'RESULTS.md'),comparison=str(run/'COMPARISON.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--wait-for',type=Path,required=True);a=p.parse_args()
    try:main(a.run,a.wait_for)
    except Exception as exc:
        write(a.run/'suite_error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
