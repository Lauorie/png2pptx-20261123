"""Isolate preview failures without flattening the delivered Office Math."""
from copy import deepcopy
import math
import subprocess

from PIL import Image, ImageDraw, ImageFont

from ..common import save
from . import equations


def runs_preview(row, folder):
    """Last-resort *preview only*. Native LaTeX/OMML is retained in the PPTX.

    A scripted-runs preview cannot express fractions/radicals. This is recorded
    as degraded, never as a successful native formula or fidelity measurement.
    No source-image pixels are copied into the reconstructed text layer.
    """
    from .layout import pillow_metrics, resolve_font
    folder.mkdir(parents=True, exist_ok=True)
    metric=pillow_metrics(row)
    width=max(1,math.ceil(metric['width']*4)+8)
    height=max(1,math.ceil(metric['height']*4)+8)
    im=Image.new('RGBA',(width,height));draw=ImageDraw.Draw(im)
    cursor=4-metric['dx']*4
    baseline=4+(24*.9-metric['dy'])*4
    for run in row['runs']:
        small=run['script']!='normal'
        info=resolve_font(row['font'],run.get('bold',row['bold']),run.get('italic',row['italic']))
        font=ImageFont.truetype(info['path'],round(24*4*(.58 if small else 1)))
        shift={'normal':0,'sub':.25,'super':-.3}[run['script']]*24*4
        draw.text((cursor,baseline+shift),run['text'],font=font,anchor='ls',
                  fill=run.get('color',row['color']))
        cursor+=font.getlength(run['text'])
    bounds=im.getbbox()
    if bounds:im=im.crop(bounds)
    asset=folder/(row['id']+'.png');im.save(asset)
    return {**metric,'width':im.width/4,'height':im.height/4,'dx':0.,'dy':0.,
            'equation_preview':str(asset.resolve()),'latex':row['latex'],
            'method':'degraded_runs_preview_native_omml_retained',
            'preview_degraded':True,'requires_review':True}


def measure_resilient(rows, folder, slots, timeout=180):
    errors={};audit=[]
    try:
        output=equations.measure(rows,folder/'standard',slots,timeout,failures=errors)
    except (ValueError,RuntimeError,OSError,TimeoutError,subprocess.TimeoutExpired) as exc:
        output={};errors={row['id']:str(exc) for row in rows}
    # Retry only failed identities. Successful previews are not regenerated.
    for index,row in enumerate(rows):
        if row['id'] in output:continue
        attempt=folder/f'recovery_{index:04d}'
        record={'id':row['id'],'original_error':errors[row['id']]}
        try:
            output.update(equations.measure([row],attempt,slots,timeout,literal_preview=True))
            record['policy']='literal_leaves_in_measurement_docx_only'
            record['native_omml_unchanged']=True
        except (ValueError,RuntimeError,OSError,TimeoutError,subprocess.TimeoutExpired) as exc:
            output[row['id']]=runs_preview(row,attempt/'runs_preview')
            record.update(policy='degraded_runs_preview_native_omml_retained',
                          recovery_error=str(exc),requires_review=True)
        audit.append(record)
        output[row['id']]['preview_recovery']=deepcopy(record)
    save(folder/'recovery_audit.json',audit)
    return output,audit
