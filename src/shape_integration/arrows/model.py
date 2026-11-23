"""Use the shared deadline/limiter; exact saved-response format feedback once."""
from ..common import read,save,sha,digest
from .prompts import SYSTEM


def ask(runtime,folder,prompt,images,validator,opt):
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'prompt.json',{'system':SYSTEM,'prompt':prompt,'images':[{'label':k,'path':str(v),'sha256':sha(v)} for k,v in images]})
    def validate(raw):
        return validate_response(raw,validator,folder)
    def repair(previous,error):
        content=previous/'content.txt';response=None
        if content.exists():response=content.read_text()
        elif (previous/'response_0.json').exists():
            data=read(previous/'response_0.json');response=data.get('choices',[{}])[0].get('message',{}).get('content')
        return {'system':SYSTEM,'prompt':{**prompt,'format_correction':{'error':error,'previous_response':response,
                'instruction':'Repair the precise invalid fields and return only one complete JSON object. No trailing Notes, explanations or second JSON, even to explain the correction. Preserve valid observations; do not restart from guesses.'}},
                'images':images,'validator':validate}
    return runtime.for_stage(shape_schema_retries=opt['protocol_retries']).call(folder/'calls','request',SYSTEM,prompt,images,validate,opt['max_tokens'],retry_builder=repair)


def validate_response(raw,validator,folder):
    """Unwrap the observed prompt-echo envelope only; keep full semantic validation."""
    fields={'objects','status','points','paths','verdict','styles','heads'}
    if isinstance(raw,dict) and not fields.intersection(raw) and isinstance(raw.get('output'),dict):
        result=validator(raw['output'])
        save(folder/'response_normalization.json',{'operation':'unwrap_output','raw_sha256':digest(raw),
             'note':'Only the JSON envelope changed. Original response and parsed_response remain in calls/.'})
        return result
    return validator(raw)
