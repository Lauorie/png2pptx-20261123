"""One persisted attempt budget, including transport and format-repair requests."""
import contextvars
import functools
import json
import os
from pathlib import Path
import threading
import time
import fcntl

ACTIVE = contextvars.ContextVar('arrow_request_budget', default=None)

class AttemptLimit(RuntimeError):
    pass

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)

def counted(function):
    @functools.wraps(function)
    def call(config, system, prompt, image, output_dir):
        state=ACTIVE.get()
        if state is not None:
            if state['used'] >= state['limit']:
                raise AttemptLimit(f"Request exhausted {state['limit']} total attempts")
            state['used']+=1
            state['calls'].append(str(Path(output_dir).resolve()))
            write(state['ledger'], {k:v for k,v in state.items() if k!='ledger'})
        return function(config, system, prompt, image, output_dir)
    return call

def bounded(function):
    @functools.wraps(function)
    def ask(self, folder, *args, **kwargs):
        limit=self.config.get('request_max_attempts')
        if limit is None or ACTIVE.get() is not None:
            return function(self, folder, *args, **kwargs)
        if type(limit) is not int or limit < 1:
            raise ValueError('request_max_attempts must be positive')
        root=Path(folder);ledger=root/'attempt_budget.json'
        old=json.loads(ledger.read_text()) if ledger.exists() else {}
        disk=len(list(root.rglob('wire_request.json')))
        state=dict(limit=limit,used=max(disk,old.get('used',0)),calls=old.get('calls',[]),ledger=ledger)
        token=ACTIVE.set(state)
        try:
            while True:
                before=state['used']
                try:return function(self, folder, *args, **kwargs)
                except Exception as exc:
                    code=getattr(exc,'status_code',None)
                    retryable=(isinstance(exc,(ValueError,TypeError,KeyError,TimeoutError,ConnectionError))
                        or getattr(exc,'retryable',False)
                        or any(t in str(exc) for t in ('TimeoutError','ReadTimeout','ConnectTimeout','RemoteProtocolError','ReadError','ConnectError','Incomplete response','Stream ended','Empty response')))
                    fatal=code in (401,403,404) or 'model_not_found' in str(exc)
                    state['last_error']=str(exc)
                    write(ledger,{k:v for k,v in state.items() if k!='ledger'})
                    if fatal or not retryable or state['used']>=limit or state['used']==before:
                        raise
                    time.sleep(self.config.get('retry_backoff_seconds',2))
        finally:ACTIVE.reset(token)
    return ask

class FileSlots:
    """Process-wide API cap across the isolated stage workers."""
    def __init__(self, folder, count):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.count=count;self.local=threading.local()

    def __enter__(self):
        offset=(os.getpid()+threading.get_ident())%self.count
        while True:
            for i in range(self.count):
                fd=os.open(self.folder/f'{(i+offset)%self.count:03d}.lock',os.O_CREAT|os.O_RDWR,0o600)
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:os.close(fd);continue
                self.local.fd=fd
                return self
            time.sleep(.05)

    def __exit__(self,*exc):
        fd=self.local.fd;fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd);del self.local.fd
