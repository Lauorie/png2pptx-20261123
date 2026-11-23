from pathlib import Path
import base64
import hashlib
import json
import threading
import time
import requests


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


class Client:
    def __init__(self, root, base_url, model, slots, thinking=True, max_tokens=18000, max_attempts=3):
        self.root = Path(root); self.base_url = base_url.rstrip('/'); self.model = model
        self.slots = slots; self.thinking = thinking; self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        if max_attempts < 1: raise ValueError('max_attempts must be positive')
        self.lock = threading.Lock()

    def call(self, tag, system, prompt, images, validator=None):
        folder = self.root / tag
        folder.mkdir(parents=True, exist_ok=False)
        content = [{'type':'text','text':json.dumps(prompt, ensure_ascii=False)}]
        records = []
        for label, path in images:
            path = Path(path); data = path.read_bytes()
            content += [{'type':'text','text':label}, {'type':'image_url','image_url':{'url':'data:image/png;base64,' + base64.b64encode(data).decode()}}]
            records.append({'label':label,'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest()})
        payload = {'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':content}],
                   'temperature':0.1,'max_tokens':self.max_tokens,'chat_template_kwargs':{'enable_thinking':self.thinking}}
        # Images are persisted separately; this manifest reconstructs the exact request.
        save(folder/'request.json', {'system':system,'prompt':prompt,'images':records,
             'parameters':{k:v for k,v in payload.items() if k!='messages'},'base_url':self.base_url})
        session = requests.Session(); session.trust_env = False
        for attempt in range(self.max_attempts):
            start=time.monotonic()
            try:
                with self.slots:
                    r=session.post(self.base_url+'/chat/completions',json=payload,timeout=(15,360))
                elapsed=time.monotonic()-start
                (folder/f'http_{attempt}.txt').write_text(r.text)
                r.raise_for_status(); raw=r.json();save(folder/f'response_{attempt}.json',raw)
                choice=raw['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Unfinished response: '+str(choice.get('finish_reason')))
                parsed=parse_json(choice['message'].get('content') or '')
                result=validator(parsed) if validator else parsed
                save(folder/'result.json',result)
                record={'tag':tag,'attempt':attempt,'seconds':round(elapsed,3),'usage':raw.get('usage'), 'success':True}
                with self.lock:
                    with (self.root/'calls.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                return result
            except Exception as exc:
                save(folder/f'error_{attempt}.json',{'type':type(exc).__name__,'error':str(exc),'seconds':time.monotonic()-start})
                if attempt + 1 == self.max_attempts: raise
                if isinstance(exc, (ValueError, KeyError, TypeError)):
                    # Same images/frame; repair the schema without sending a long conversation.
                    payload['messages'][1]['content'][0]['text']=json.dumps(prompt,ensure_ascii=False)+'\nYour previous response failed validation: '+str(exc)+'\nReturn a complete corrected JSON object.'
                    save(folder/f'corrected_prompt_{attempt+1}.json', {'prompt':payload['messages'][1]['content'][0]['text']})
                time.sleep(1+attempt)
        raise RuntimeError('unreachable')
