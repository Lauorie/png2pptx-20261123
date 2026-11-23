"""CPU rjob owner for a fresh complete PPT batch, hosted GLM and shared SAM."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_recovery.api_gateway import ApiRelay
from full_recovery.api_budget import BudgetGuard, BudgetPaused


def read(path):return json.loads(Path(path).read_text())
def read_progress(path, previous=None):
    """Progress is telemetry; an incomplete legacy write must not kill work."""
    try:
        value=read(path)
        if not isinstance(value,dict):raise ValueError('Progress must be an object')
        return value,None
    except (FileNotFoundError,json.JSONDecodeError,ValueError) as exc:
        return previous or {},type(exc).__name__

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main(*, pipeline_argv=None):
    Path('/proc/self/oom_score_adj').write_text('0')
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    run=p.parse_args().run_dir.resolve();settings=read(run/'settings.json')
    shape_only=settings.get('workflow_scope')=='shape'
    merge_only=settings.get('workflow_scope')=='composition'
    text_only=settings.get('workflow_scope')=='text'
    branches_only=settings.get('workflow_scope')=='shape_text'
    shape_work=Path(settings.get('shape_work_root',str(run/'shape')))
    owner=(run/'worker.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=run/'code';shape=run/'shape/code';project=Path(settings['source_project'])
    started=time.time();children=[];handles=[];servers=[];phase='preflight'
    budget=BudgetGuard(settings['api_budget_config']) if settings.get('api_budget_config') else None
    def stamp(name,**extra):
        nonlocal phase
        phase=name;row={'state':name,'host':socket.gethostname(),'pid':os.getpid(),
            'at':time.time(),'elapsed_seconds':time.time()-started,'gpu':0,
            'workflow_scope':settings.get('workflow_scope'),**extra}
        save(run/'status.json',row)
        if settings.get('shape_base_run'):
            save(Path(settings['shape_base_run'])/'shape/status.json',{**row,'resume_snapshot':str(run)})
        print(name,extra,flush=True)
    def stop(*args):
        for child in reversed(children):
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for server in servers:server.shutdown()
        if args:
            stamp('stopped');raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,argv,env,cwd=code):
        handle=(run/'logs'/(name+'.log')).open('a',buffering=1);handles.append(handle)
        child=subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdout=handle,
            stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    try:
        if budget:budget.check()
        stamp('verifying_frozen_code_and_original_inputs')
        for prefix,manifest in [(run,run/'code_manifest.json'),(shape,run/'shape/code_sha256.json')]:
            for relative,digest in read(manifest).items():
                if hashlib.sha256((prefix/relative).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Frozen code changed: '+relative)
        originals=read(run/'source_manifest.json')['images']
        expected_work=len(originals)
        if branches_only:
            selection=read(run/'resume_selection.json')
            expected_work=len(selection['pending_shape'])+len(selection['pending_text'])
        for item in originals:
            if hashlib.sha256(Path(item['original']).read_bytes()).hexdigest()!=item['source_sha256']:
                raise ValueError('Original input changed: '+item['id'])
        stamp('renderer_setup')
        sys.path.insert(0,str(shape/'scripts'))
        from shape_worker import renderer
        native=renderer(project,shape);portable=Path('/tmp/shape-recovery-lo')
        from full_recovery.fonts import configure
        save(run/'font_validation.json',configure(run/'backends/reference_fonts',portable))
        raw=portable/'bin/libreoffice-raw'
        shutil.copy2(run/'renderer_runtime/lo_cpu.py',raw);raw.chmod(0o755)
        if settings.get('native_renderer'):
            raw=Path(settings['native_renderer'])
            if not raw.is_file():raise FileNotFoundError('Native renderer absent: '+str(raw))
        guarded=portable/'bin/libreoffice';shutil.copy2(code/'lo_guard.py',guarded);guarded.chmod(0o755)
        system=Path('/usr/bin/libreoffice')
        if system.exists() or system.is_symlink():system.unlink()
        system.symlink_to(guarded)
        env={**os.environ,**native,'STYLE_CLUSTER_WORKER':'1','PPT_LOCAL_GLM':'1',
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','PYTHONNOUSERSITE':'1',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'PYTHONPATH':os.pathsep.join(map(str,[shape/'src',shape/'vendor',project/'.runtime/shape_layers_py311'])),
            'PPT_CPU_GLIBC_DIR':str(run/'renderer_runtime/glibc235'),
            'PPT_RENDER_LOCK_DIR':str(run/'render_locks'),'PPT_LIBREOFFICE_RAW':str(raw),
            'PPT_SHARED_SAM_LOCK_ROOT':str(project/'.runtime/sam_endpoint_locks'),
            'FONTCONFIG_FILE':str(portable/'fonts.conf'),
            'LD_LIBRARY_PATH':'/opt/shape-cpu/lib:'+os.environ.get('LD_LIBRARY_PATH','')}
        env['PATH']=str(portable/'bin')+':/opt/shape-cpu/bin:'+os.environ['PATH']
        if settings.get('api_force_reasoning_effort'):
            env['PPT_FORCE_REASONING_EFFORT']=settings['api_force_reasoning_effort']
        if settings.get('native_renderer'):
            result=subprocess.run([str(raw),'--headless','--version'],env=env,
                capture_output=True,text=True,timeout=30,check=True)
            save(run/'renderer_validation.json',{'executable':str(raw),'version':result.stdout.strip()})
        if not shape_only and not text_only and not merge_only and not branches_only and settings.get('arrow_driver')=='frozen_geometry_first':
            # The geometry-first arrow critic was validated with 26.8. Keep
            # that renderer contract; never silently substitute the 7.x text
            # renderer, which can alter endpoint-head evidence.
            arrow_renderer=Path(settings.get('arrow_native_renderer',str(project/'.runtime/arrow_renderer/soffice')))
            if not arrow_renderer.is_file():
                raise FileNotFoundError('Arrow renderer 26.8 is absent: '+str(arrow_renderer))
            env['ARROW_LIBREOFFICE']=str(arrow_renderer)
            version=subprocess.run([str(arrow_renderer),'--headless','--version'],capture_output=True,text=True,timeout=20,check=True).stdout.strip()
            if not version.startswith('LibreOffice 26.8.'):
                raise ValueError('Arrow renderer mismatch: '+version)
            save(run/'arrow_renderer_validation.json',{'executable':str(arrow_renderer),'version':version})
            from full_recovery.renderer_probe import validate as validate_native_render
            previous_env = dict(os.environ)
            try:
                os.environ.update(env)
                validate_native_render(arrow_renderer, run/'renderer_smoke'/str(time.time_ns()))
            finally:
                os.environ.clear(); os.environ.update(previous_env)
        if shape_only:env['GLM_API_RELAY_URL']=settings['glm_url']
        # Code-change tests run once before freezing the job. Production only
        # checks its runtime and exact inputs; do not replay the whole test suite.
        save(run/'preflight_policy.json',{'runtime_only':True,
            'unit_tests':'targeted checks run separately before the freeze'})
        direct_opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        api_opener=urllib.request.build_opener()
        if not merge_only and not text_only:
            stamp('checking_shared_sam')
            with direct_opener.open(settings['sam_url'].rstrip('/')+'/health',timeout=15) as response:health=json.load(response)
            model=health.get('model',{});checkpoint=model.get('checkpoint',{})
            if health.get('phase') not in ('ready','busy') or checkpoint.get('epoch')!=settings['external_sam_epoch'] or model.get('checkpoint_sha256')!=settings['external_sam_checkpoint_sha256']:
                raise ValueError('Shared SAM is not ready or its checkpoint identity differs')
            save(run/'sam_validation.json',{'at':time.time(),'model':model,'shared_service':True,'url':settings['sam_url']})
        relays=settings.get('api_relays',[{'role':'shared','port':settings['api_gateway_port']}])
        for spec in relays:
            metrics=run if len(relays)==1 else run/'api'/spec['role']
            metrics.mkdir(parents=True,exist_ok=True)
            relay=ApiRelay({**settings,**spec},metrics);server=relay.server(spec['port'])
            servers.append(server)
            threading.Thread(target=server.serve_forever,daemon=True).start();relay.stamp()
        stamp('checking_hosted_glm_api')
        probe={'model':settings['api_model'],'max_tokens':256,'temperature':0,
               'chat_template_kwargs':{'reasoning_effort':settings.get('api_force_reasoning_effort','low')},
               'messages':[{'role':'user','content':'Return exactly this JSON: {"ready":true}'}]}
        probe_timeout=180
        request=urllib.request.Request(settings['glm_url']+'/chat/completions',json.dumps(probe).encode(),
            {'Content-Type':'application/json','X-PPT-Request-Timeout-Seconds':str(probe_timeout)})
        with api_opener.open(request,timeout=probe_timeout+10) as response:result=json.load(response)
        finish=result['choices'][0]['finish_reason']
        if finish not in ('stop','length'):raise ValueError('API preflight transport failed')
        save(run/'api_validation.json',{'at':time.time(),'model':result.get('model'),'usage':result.get('usage'),
             'requested_service_tier':settings.get('api_service_tier'),
             'returned_service_tier':result.get('service_tier'),
             'probe_finish_reason':finish,'probe_scope':'transport readiness only; actual task responses still require complete validated output',
             'transport':'hosted SSE; JSON assembled for shape; live SSE for reference text/arrow',
             'global_concurrency':settings['api_global_concurrency'],'credential_in_logs':False})
        phase_name='recovering_shape_and_text' if branches_only else 'merging_existing_products' if merge_only else 'recovering_shapes' if shape_only else 'recovering_text' if text_only else 'recovering_full_ppt'
        work_root=shape_work if shape_only else run
        entry=code/'branches_only.py' if branches_only else code/'merge_existing.py' if merge_only else shape/'run_shapes.py' if shape_only else code/'text_only.py' if text_only else code/'run_ppt.py'
        if shape_only and settings.get('shape_work_root'):
            from full_recovery.advance_shape import activate
            stamp('activating_unfinished_shape_checkpoints')
            activate(run)
        stamp(phase_name,completed=0,failed=0,expected=expected_work)
        child=launch('pipeline',[sys.executable,'-B','-u',*(pipeline_argv or [entry,'--run-dir',work_root])],env)
        progress={}
        while child.poll() is None:
            if budget:save(run/'budget_status.json',budget.check())
            if time.time()-started>settings['worker_max_seconds']:
                raise TimeoutError('Full API batch lifetime exceeded; checkpoints retained')
            progress,progress_warning=read_progress(work_root/'progress.json',progress)
            completed=(progress.get('done',0)-progress.get('failed',0)) if shape_only else progress.get('completed',0)
            stamp(phase_name,completed=completed,failed=progress.get('failed',0),expected=expected_work,
                  **({'progress_read_warning':progress_warning} if progress_warning else {}))
            time.sleep(2 if budget else 10)
        if child.returncode:raise RuntimeError('Full API pipeline failed; inspect per-stage checkpoints')
        summary=read(work_root/'summary.json')
        stamp('completed_with_pending_arrow_inputs' if summary.get('pending_arrow_ids') else 'completed',
              exported=summary['exported'],pending_arrow_ids=summary.get('pending_arrow_ids',[]))
    except BudgetPaused as exc:
        stamp('paused_budget',reason=str(exc),checkpoints_retained=True)
        return
    except BaseException as exc:
        if budget and budget.pause_path.exists():
            stamp('paused_budget',reason=read(budget.pause_path)['reason'],checkpoints_retained=True)
            return
        stamp('failed',error=str(exc));raise
    finally:
        stop()
        for handle in handles:handle.close()


if __name__=='__main__':main()
