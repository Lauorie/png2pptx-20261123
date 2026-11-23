"""Conservative source-pixel color recovery using measured glyph positions.

Does not OCR, change text, invent math, move boxes, or sample an entire filled
rectangle as if it were a font color. Uncertain glyphs retain model colors.
"""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import math
from pathlib import Path
import unicodedata

import numpy as np
from PIL import Image
from ..resource_limits import bounded


def _hex(rgb):
    return '#%02X%02X%02X' % tuple(int(v) for v in rgb)


def _background(image, box):
    x0,y0,x1,y1=box
    x0=max(0,int(math.floor(x0))-3);y0=max(0,int(math.floor(y0))-3)
    x1=min(image.shape[1],int(math.ceil(x1))+4);y1=min(image.shape[0],int(math.ceil(y1))+4)
    roi=image[y0:y1,x0:x1]
    if not roi.size:return np.array([255,255,255])
    border=np.concatenate((roi[:2].reshape(-1,3),roi[-2:].reshape(-1,3),roi[:,:2].reshape(-1,3),roi[:,-2:].reshape(-1,3)))
    quantized=(border//8).astype(int)
    keys,count=np.unique(quantized,axis=0,return_counts=True)
    key=keys[count.argmax()]
    return np.median(border[np.all(quantized==key,axis=1)],axis=0)


def glyph_color(image, row, glyph, background):
    points=np.asarray(glyph.get('ink_points',[]),dtype=float)
    if not len(points):return None,{'reason':'no_measured_ink'}
    measurement=row['layout']['measurement'];factor=row['font_size_px']/measurement['reference_size_px']
    x,y,w,h=row['frame_px'];left,top=row.get('text_insets_px',[0,0])[:2]
    a=math.radians(row['rotation_deg']);c,s=math.cos(a),math.sin(a)
    local=points*factor+np.array([left-w/2,top-h/2])
    world=local@np.array([[c,s],[-s,c]])+np.array([x+w/2,y+h/2])
    # Small alignment tolerance, only around predicted glyph ink, not the bbox.
    offsets=np.array([[dx,dy] for dx in (-1,0,1) for dy in (-1,0,1)])
    xy=np.rint(world[:,None,:]+offsets[None,:,:]).astype(int).reshape(-1,2)
    xy=np.unique(xy,axis=0)
    xy=xy[(xy[:,0]>=0)&(xy[:,0]<image.shape[1])&(xy[:,1]>=0)&(xy[:,1]<image.shape[0])]
    if not len(xy):return None,{'reason':'outside_source'}
    pixels=image[xy[:,1],xy[:,0]]
    contrast=np.linalg.norm(pixels.astype(float)-background,axis=1)
    foreground=pixels[contrast>max(32.,float(np.percentile(contrast,85))*.65)]
    if len(foreground)<4:return None,{'reason':'insufficient_foreground'}
    quantized=(foreground//8).astype(int)
    colors,counts=np.unique(quantized,axis=0,return_counts=True)
    index=int(counts.argmax());fraction=float(counts[index]/len(foreground))
    if fraction<.25:return None,{'reason':'ambiguous_color','peak_fraction':fraction}
    rgb=np.median(foreground[np.all(quantized==colors[index],axis=1)],axis=0).round().astype(int)
    # Avoid manufacturing many barely different antialias colors for black ink.
    if rgb.max()<24:rgb[:]=0
    if rgb.min()>239:rgb[:]=255
    return _hex(rgb),{'method':'source_pixels_at_native_glyph_ink','peak_fraction':fraction,'samples':len(xy),
                      'foreground_fraction':float(len(foreground)/len(xy))}


def recover_formula(row, image, folder):
    """Calibrate a uniform formula without flattening or altering its structure."""
    if row.get('layout',{}).get('measurement',{}).get('preview_degraded'):
        return row,{'policy':'degraded_preview_no_color_calibration'}
    from .omml import latex_to_omml
    from .equations import NS
    tree=latex_to_omml(row['latex'],color=row['color'])
    colors=set(tree.xpath('.//a:srgbClr/@val',namespaces=NS))
    if len(colors)>1 or 'textcolor' in row['latex'] or '\\color' in row['latex']:
        return row,{'policy':'structured_multicolor_formula_preserved'}
    quality=row.get('glyph_registration',{})
    if quality.get('status')!='accepted' or quality.get('score',0)<.70:
        return row,{'policy':'formula_color_unverified_geometry'}
    metric=row['layout']['measurement'];path=metric.get('equation_preview')
    if not path or folder is None:return row,{'policy':'formula_color_no_output_asset'}
    with Image.open(path) as im:rgba=np.array(im.convert('RGBA'))
    yy,xx=np.where(rgba[:,:,3]>180)
    if len(xx)<8:return row,{'policy':'formula_color_insufficient_ink'}
    step=max(1,len(xx)//2048)
    points=np.column_stack(((xx[::step]+.5)/rgba.shape[1]*metric['width'],
                            (yy[::step]+.5)/rgba.shape[0]*metric['height'])).tolist()
    background=_background(image,row['bbox_2d'])
    measured,evidence=glyph_color(image,row,{'ink_points':points},background)
    if measured is None:return row,{'policy':'formula_color_ambiguous',**evidence}
    old=row['color'];row['color']=measured
    for run in row['runs']:run['color']=measured
    # Update only the independently rendered compatibility asset. Source image
    # pixels are never copied, and the delivered PPT still contains native OMML.
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rgba[:,:,:3]=[int(measured[i:i+2],16) for i in (1,3,5)]
    output=folder/(row['id']+'_equation.png');Image.fromarray(rgba).save(output)
    metric['equation_preview']=str(output.resolve())
    return row,{'policy':'pixel_calibrated_uniform_formula','before':old,'after':measured,**evidence}


def run_consensus(row, samples):
    """Share strong color evidence only with agreeing glyphs in the same run.

    A consensus cannot recolor a differently colored or unmeasured character.
    It only relaxes peak purity for antialiased glyphs after accepted geometric
    registration and at least two independent high-purity glyph observations.
    """
    quality=row.get('glyph_registration',{})
    if quality.get('status')!='accepted' or quality.get('score',0)<.80:return {}
    result={};offset=0
    rgb=lambda color:np.array([int(color[i:i+2],16) for i in (1,3,5)])
    for run in row['runs']:
        ids=[offset+i for i,c in enumerate(run['text']) if not c.isspace()]
        offset+=len(run['text'])
        measured=[i for i in ids if samples.get(i,(None,{}))[0] is not None]
        strong=[i for i in measured if samples[i][1].get('peak_fraction',0)>=.50
                and samples[i][1].get('foreground_fraction',0)>=.15]
        if len(strong)<2 or len(strong)<len(ids)*.35:continue
        key=max(strong,key=lambda i:sum(np.linalg.norm(rgb(samples[i][0])-rgb(samples[j][0]))<=24 for j in strong))
        color=samples[key][0]
        agreeing=[i for i in measured if np.linalg.norm(rgb(color)-rgb(samples[i][0]))<=24]
        if len(agreeing)<len(measured)*.80:continue
        for i in agreeing:
            evidence=samples[i][1]
            if evidence.get('peak_fraction',0)>=.25 and evidence.get('foreground_fraction',0)>=.15:
                result[i]={'policy':'same_run_agreeing_glyphs','strong_glyphs':len(strong),
                           'agreeing_glyphs':len(agreeing),'run_glyphs':len(ids)}
    return result


@bounded('cpu')
def recover(rows, source, folder=None):
    with Image.open(source) as im:image=np.asarray(im.convert('RGB'))
    output,report=[],[]
    for original in rows:
        row=deepcopy(original)
        if row.get('kind')=='formula':
            row,evidence=recover_formula(row,image,folder)
            output.append(row);report.append({'id':row['id'],**evidence});continue
        glyphs=row.get('layout',{}).get('measurement',{}).get('characters',[])
        expected=[];styles=[]
        for run in row['runs']:
            expected.extend((char,run['script'],run.get('color',row['color'])) for char in run['text'])
            # Recoloring must not flatten bold/italic/underline boundaries.
            styles.extend({k:run[k] for k in ('bold','italic','underline') if k in run}
                          for char in run['text'])
        normalize=lambda value:unicodedata.normalize('NFKC',value)
        matcher=SequenceMatcher(None,[normalize(v[0]) for v in expected],[normalize(g['text']) for g in glyphs],autojunk=False)
        matches={i+k:j+k for i,j,n in matcher.get_matching_blocks() for k in range(n)}
        background=_background(image,row['bbox_2d']);runs=[];changes=[];uncertain=[]
        samples={i:glyph_color(image,row,glyphs[matches[i]],background)
                 for i,(char,_,_) in enumerate(expected) if i in matches and not char.isspace()}
        consensus=run_consensus(row,samples)
        for index,(char,script,old) in enumerate(expected):
            color=old
            if index in matches and not char.isspace():
                measured,evidence=samples[index]
                if index in consensus:evidence={**evidence,'consensus':consensus[index]}
                if measured is not None:
                    reference=np.array([int(old[i:i+2],16) for i in (1,3,5)])
                    candidate=np.array([int(measured[i:i+2],16) for i in (1,3,5)])
                    # Geometry remains approximate. A neighboring fill/line
                    # must not recolor a glyph (e.g. black Mask -> white grid).
                    # Pixel calibration refines VLM run colors, not their hue.
                    quality=row.get('glyph_registration',{})
                    verified=(quality.get('status')=='accepted' and quality.get('score',0)>=.80
                              and evidence.get('peak_fraction',0)>=.50
                              and evidence.get('foreground_fraction',0)>=.15)
                    if np.linalg.norm(candidate-reference)>80 and not (verified or index in consensus):
                        measured=None;evidence={**evidence,'reason':'conflicts_with_semantic_run_color'}
                if measured is not None:
                    color=measured
                    if measured!=old:changes.append({'index':index,'char':char,'before':old,'after':measured,**evidence})
                else:uncertain.append({'index':index,'char':char,**evidence})
            style=styles[index]
            if (runs and runs[-1]['script']==script and runs[-1]['color']==color
                    and {k:runs[-1][k] for k in ('bold','italic','underline') if k in runs[-1]}==style):
                runs[-1]['text']+=char
            else:runs.append({'text':char,'script':script,'color':color,**style})
        if ''.join(r['text'] for r in runs)!=row['text']:raise ValueError('Color recovery changed text')
        row['runs']=runs
        colors=Counter()
        for run in runs:
            colors[run['color']]+=len(run['text'].strip())
        if colors:row['color']=colors.most_common(1)[0][0]
        row['color_recovery']={'matched_glyphs':len(matches),'total_characters':len(expected),'changed_characters':len(changes)}
        report.append({'id':row['id'],'text':row['text'],'background':_hex(background.round()),
                       'changes':changes,'uncertain':uncertain,'matched_glyphs':len(matches),'total_characters':len(expected)})
        output.append(row)
    return output,report
