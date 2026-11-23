"""Small style-only requests for existing geometry; no identities or coordinates."""
from pathlib import Path
import numpy as np
from PIL import ImageDraw
import protocol,visual
from transport.common import write_json as write,sha256


def prompt(rows):
    return dict(task='Classify ONLY the stroke style of each named existing path from the original image.',
        line_style_types=protocol.LINE_STYLES,
        targets=[dict(id=f'L{i}',kind=r['kind'],start=np.round(r['start'],1).tolist(),end=np.round(r['end'],1).tolist()) for i,r in enumerate(rows,1)],
        instructions=['CLEAN is the original crop. LOCATOR is the same crop with thin colored guides and short L labels; guides locate targets, their colors and continuity are not evidence of style.',
            'Return solid, dashed, dotted or unknown for each L. Look at the visible shaft, excluding the arrowhead. Occlusion and SAM gaps alone do not make a stroke dashed.',
            'For a fragment inside an arrowhead, use the connected visible shaft if its identity is clear; otherwise unknown. If the proposed path has no identifiable original stroke, use unknown. Do not edit geometry, heads or identities, and do not estimate any dimensions.'],
        output_schema={'line_styles':'object mapping every supplied L ID to solid/dashed/dotted/unknown'})


def locator(clean,rows,path):
    im=clean.copy();d=ImageDraw.Draw(im)
    for i,r in enumerate(rows,1):
        points=np.array([r['start'],*r.get('via',[]),r['end']])
        if r.get('beziers'):
            points=[];t=np.linspace(0,1,24)[:,None]
            for a,b,c,e in np.array(r['beziers']):points.extend(((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*e).tolist())
            points=np.array(points)
        color=visual.color(i-1);d.line(list(map(tuple,points)),fill=color,width=1)
        visual.label(d,points[len(points)//2],f'L{i}',color,im)
    im.save(path)


def classify(clean,rows,folder,model):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    clean.save(folder/'clean.png');locator(clean,rows,folder/'locator.png')
    ids=[f'L{i}' for i in range(1,len(rows)+1)]
    write(folder/'mapping.json',{k:r['id'] for k,r in zip(ids,rows)})
    def validate(a):protocol.line_styles(a,ids);return a
    images=[('CLEAN: original image, ground truth for visible stroke',folder/'clean.png'),
            ('LOCATOR: thin geometry guides and L labels, not reconstructed styles',folder/'locator.png')]
    write(folder/'images.json',[dict(label=k,path=str(p.resolve()),sha256=sha256(p)) for k,p in images])
    answer=model.ask(folder/'request',prompt(rows),images,validate)
    return {r['id']:answer['line_styles'][k] for k,r in zip(ids,rows)}
