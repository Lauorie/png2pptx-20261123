"""OpenAI-compatible transport; optional relay credentials are never persisted."""
from pathlib import Path
import base64
import hashlib
import json
import re
import shlex
import threading
import time
import requests
from .client import save, parse_json


def load_bashrc_credentials(path=None):
    path=Path(path or Path.home()/'.bashrc')
    names={'ANTHROPIC_BASE_URL','ANTHROPIC_AUTH_TOKEN','CODEX_API_KEY','OPENAI_API_KEY',
           'OPENAI_BASE_URL','OPENAI_API_BASE','BOYUE_BASE_URL','BOYUE_API_KEY','BOYUE_API_BASE'}
    values={};line_numbers={}
    for i,line in enumerate(path.read_text().splitlines(),1):
        m=re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$',line)
        if not m or m[1] not in names:continue
        # Parse simple literals/variable aliases only; never source arbitrary shell code.
        if '`' in m[2] or '$(' in m[2]:continue
        tokens=shlex.split(m[2],comments=True)
        if len(tokens)!=1:continue
        value=re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)',
                     lambda n:values.get(n[1] or n[2],''),tokens[0])
        values[m[1]]=value;line_numbers[m[1]]=i
    base_name=next((k for k in ['BOYUE_BASE_URL','BOYUE_API_BASE','OPENAI_BASE_URL','OPENAI_API_BASE','ANTHROPIC_BASE_URL'] if values.get(k)),None)
    key_name=next((k for k in ['BOYUE_API_KEY','OPENAI_API_KEY','CODEX_API_KEY','ANTHROPIC_AUTH_TOKEN'] if values.get(k)),None)
    if not base_name or not key_name:raise ValueError('Could not read relay URL and key from simple bashrc assignments')
    base=values[base_name].rstrip('/')
    if not base.endswith('/v1'):base+='/v1'
    return base,values[key_name],{'source_path':str(path.resolve()),'base_url_variable':base_name,
        'base_url_line':line_numbers[base_name],'key_variable':key_name,'key_line':line_numbers[key_name]}


class RelayClient:
    def __init__(self,root,model,slots,base_url,key,max_tokens=12000,max_attempts=2,read_timeout=360,thinking='default',thinking_off_strategy='empty_think_prefix'):
        self.root=Path(root);self.model=model;self.slots=slots;self.base_url=base_url
        self._key=key;self.max_tokens=max_tokens;self.lock=threading.Lock()
        self.max_attempts=max_attempts
        if max_attempts<1:raise ValueError('max_attempts must be positive')
        self.read_timeout=read_timeout
        if read_timeout<=0:raise ValueError('read_timeout must be positive')
        if thinking not in ('default','on','off'):raise ValueError('thinking must be default, on or off')
        self.thinking=thinking
        if thinking_off_strategy not in ('template_flag','empty_think_prefix'):raise ValueError('Unknown thinking-off strategy')
        self.thinking_off_strategy=thinking_off_strategy

    def request_parameters(self):
        parameters={'model':self.model,'temperature':0.1,'max_tokens':self.max_tokens}
        if self.thinking!='default':
            parameters['chat_template_kwargs']={'enable_thinking':self.thinking=='on'}
        if self.thinking=='off' and self.thinking_off_strategy=='empty_think_prefix':
            parameters.update(add_generation_prompt=False,continue_final_message=True)
        return parameters

    def request_messages(self,system,content):
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        if self.thinking=='off' and self.thinking_off_strategy=='empty_think_prefix':
            messages.append({'role':'assistant','content':'<think></think>'})
        return messages

    def redact(self,value):
        return str(value).replace(self._key,'<REDACTED>') if self._key else str(value)

    def call(self,tag,system,prompt,images,validator=None):
        folder=self.root/tag;folder.mkdir(parents=True,exist_ok=False)
        content=[{'type':'text','text':json.dumps(prompt,ensure_ascii=False)}];records=[]
        for label,path in images:
            path=Path(path);data=path.read_bytes()
            content += [{'type':'text','text':label},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}]
            records.append({'label':label,'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest()})
        payload={**self.request_parameters(),'messages':self.request_messages(system,content)}
        session=requests.Session();session.trust_env=False
        if self._key:
            session.headers.update({'Authorization':'Bearer '+self._key})
        save(folder/'request.json',{'system':system,'prompt':prompt,'images':records,
             'parameters':{k:v for k,v in payload.items() if k!='messages'},'base_url':self.base_url,
             'transport':{'connect_timeout_seconds':15,'read_timeout_seconds':self.read_timeout,'max_attempts':self.max_attempts},
             'auth':'Bearer key read from bashrc; value deliberately omitted' if self._key else 'none; direct local endpoint'})
        for attempt in range(self.max_attempts):
            start=time.monotonic()
            try:
                with self.slots:r=session.post(self.base_url+'/chat/completions',json=payload,timeout=(15,self.read_timeout))
                body=self.redact(r.text);(folder/f'http_{attempt}.txt').write_text(body)
                r.raise_for_status();raw=json.loads(body);save(folder/f'response_{attempt}.json',raw)
                choice=raw['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('Unfinished response '+str(choice.get('finish_reason')))
                parsed=parse_json(choice['message'].get('content') or '')
                result=validator(parsed) if validator else parsed;save(folder/'result.json',result)
                event={'tag':tag,'attempt':attempt,'seconds':time.monotonic()-start,'success':True,'usage':raw.get('usage'),'response_model':raw.get('model')}
                with self.lock:
                    with (self.root/'calls.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                return result
            except Exception as exc:
                error=self.redact(exc);save(folder/f'error_{attempt}.json',{'type':type(exc).__name__,'error':error,'seconds':time.monotonic()-start})
                if isinstance(exc,requests.HTTPError) and exc.response.status_code<500:raise RuntimeError(error) from None
                if attempt+1==self.max_attempts:raise RuntimeError(error) from None
                if isinstance(exc,(ValueError,KeyError,TypeError)):
                    payload['messages'][1]['content'][0]['text']=json.dumps(prompt,ensure_ascii=False)+'\nReturn a complete corrected JSON object. Validation error: '+error
                    save(folder/f'corrected_prompt_{attempt+1}.json',{'prompt':payload['messages'][1]['content'][0]['text']})
                time.sleep(1)
        raise RuntimeError('unreachable')
