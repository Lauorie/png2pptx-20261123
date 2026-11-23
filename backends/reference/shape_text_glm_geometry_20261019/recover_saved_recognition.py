"""Explicit offline salvage of individually valid items from a completed OCR response.

Never clips coordinates, invents text, or declares partial recovery complete.
Original responses are immutable; rejected items remain in the derived audit.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha
from shape_integration.texts import contracts,options
from shape_integration.texts.controller import merge


def recover(raw,size):
    if not isinstance(raw,dict) or not isinstance(raw.get('texts'),list):
        raise ValueError('Expected a complete texts response')
    accepted=[];rejected=[]
    for index,item in enumerate(raw['texts']):
        try:
            accepted.extend(contracts.recognition({'texts':[item]},size)['texts'])
        except ValueError as exc:
            rejected.append({'response_index':index,'proposal':item,'error':str(exc)})
    if not accepted:raise ValueError('No individually valid items')
    return accepted,rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--response',type=Path,required=True)
    p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    status=read(a.response.parent/'stream_status.json')
    if status.get('finish_reason')!='stop' or not status.get('sse_done') or status.get('http_status')!=200:
        raise ValueError('Only complete successful HTTP/SSE responses may be recovered')
    if a.output.exists():raise ValueError('Use a fresh output')
    frame=read(a.frame)
    with Image.open(a.image) as im:
        if list(im.size)!=frame['source_size']:raise ValueError('Source dimensions do not match frame')
    raw=read(a.response);accepted,rejected=recover(raw,frame['size'])
    rows=[];history=[]
    merge(rows,accepted,frame,'validated_saved_response',history,options({}))
    a.output.mkdir(parents=True)
    errors=[f'{len(rejected)} rejected OCR proposals retained in recovery_audit.json',
            'Saved response recovery only; no local content/geometry revalidation']
    save(a.output/'recognition.json',{'texts':rows,'uncertain':raw.get('uncertain',[]),'errors':errors})
    save(a.output/'recovery_audit.json',{'state':'partial_recovery','api_calls':0,
         'accepted':len(accepted),'rejected':rejected,'merge_history':history,
         'provenance':{str(f.resolve()):sha(f) for f in (a.response,a.frame,a.image)},
         'visual_accuracy_verified':False})
    print(f'Recovered {len(rows)} items; rejected {len(rejected)}; local review still required',flush=True)


if __name__=='__main__':main()
