"""Preserved arrow batch entry, with a local GLM transport adapter."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',type=Path,required=True);p.add_argument('--test',action='store_true')
    args,rest=p.parse_known_args();root=args.backend.resolve();sys.path[:0]=[str(root),str(root/'lib')]
    if args.test:
        import pytest
        raise SystemExit(pytest.main([str(root),'-q','--disable-warnings',
            '--basetemp',str(root.parents[1]/'tests_tmp/arrow')]))
    from transport import credentials,client
    # Both branches share one external SAM service. Serialize its existing
    # segment call with the exact lock used by the shape Models adapter.
    if os.environ.get('PPT_SHARED_SAM_LOCK_ROOT'):
        import fcntl
        import hashlib
        from sam_client import SamClient
        original_segment=SamClient.segment
        def shared_segment(self,*a,**kw):
            folder=Path(os.environ['PPT_SHARED_SAM_LOCK_ROOT']);folder.mkdir(parents=True,exist_ok=True)
            key=hashlib.sha256(self.url.rstrip('/').encode()).hexdigest()
            with (folder/key).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                return original_segment(self,*a,**kw)
        SamClient.segment=shared_segment
    # The remote transport redacts its credential in streamed evidence. A
    # non-secret sentinel keeps that existing redaction code unchanged; the
    # local GLM endpoint does not enforce authentication.
    def local_credentials(base):
        return base.rstrip('/'),'local-glm-without-auth',{'source':'local GLM; authentication disabled','key_saved':False}
    credentials.load_credentials=local_credentials;client.load_credentials=local_credentials
    original=client.build_payload
    def local_payload(*a,**kw):
        payload=original(*a,**kw);thinking=payload.pop('thinking',None);effort=payload.pop('reasoning_effort',None)
        template={'clear_thinking':True}
        if thinking is not None:template['enable_thinking']=thinking.get('type')=='enabled'
        if effort is not None:template['reasoning_effort']=effort
        payload['chat_template_kwargs']=template
        return payload
    client.build_payload=local_payload
    # Style estimation has a compact contract. Preserve geometry's original
    # budget, but keep contradictory dash evidence from consuming it in style.
    import style_pipeline
    from arrow_finalize import profile
    original_recover=style_pipeline.recover
    def bounded_style(crop,rows,supports,out,config,model,locks,*a,**kw):
        styled=profile(model,max_tokens=config.get('style_max_tokens',8192),
            reasoning_effort=config.get('style_reasoning_effort','low'),
            request_timeout_seconds=config.get('style_request_timeout_seconds',360),
            transport_retries=config.get('style_transport_retries',1))
        return original_recover(crop,rows,supports,out,config,styled,locks,*a,**kw)
    style_pipeline.recover=bounded_style
    sys.argv=[str(root/'run.py'),*rest]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
