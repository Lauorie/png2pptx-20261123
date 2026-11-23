"""Semantic shape review: one current block arrow, tolerant of contour deviations."""
import copy
from pathlib import Path
import cairosvg
from PIL import Image,ImageDraw
import block_shapes,shape_render
from transport.common import read_json as read,write_json as write

def view(crop,row,folder,observation=None,color='#1750d0'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    styled=copy.deepcopy(row)
    styled['style']=dict(color=color,shape_parts={p['id']:dict(fill=None,stroke=color,width_px=2,dash='solid') for p in row['shape_parts']})
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{crop.width}" height="{crop.height}" viewBox="0 0 {crop.width} {crop.height}">'+shape_render.svg_parts(styled)+'</svg>'
    (folder/'outline.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(folder/'outline_rgba.png'))
    image=crop.convert('RGBA');image.alpha_composite(Image.open(folder/'outline_rgba.png').convert('RGBA'))
    draw=ImageDraw.Draw(image)
    if observation:
        for point in observation.get('points',[]):
            x,y=point['xy'];draw.ellipse([x-3,y-3,x+3,y+3],outline=color,width=1)
    image.convert('RGB').save(folder/'overlay.png')
    return folder/'overlay.png'

VERSION='block_shape_semantic_critic.2'

def prompt(can_refine):
    choices=['accept','unresolved']+(['refine'] if can_refine else [])
    return dict(protocol=VERSION,
        task='Does CURRENT convey the same block arrow as CLEAN? Judge its meaning, not pixel-level contour accuracy.',
        images=['CLEAN: original crop.', 'CURRENT: fitted boundary over the same crop. Blue outlines and hollow point markers are annotations.'],
        accept='Accept when the head and tail occupy the same approximate regions, the arrow points the same way, and its broad bend/continuity expresses the same arrow. For complex curved block arrows, differences in contour position, curvature, thickness, taper and small notches are acceptable when this meaning is preserved. Do not request refinement just to improve these details.',
        correct='Only flag a clear semantic error: wrong direction, head/tail in the wrong region, missing head or substantial body, a broken arrow, or a different overall route. Ignore color, fill, stroke style and point markers. Do not calculate radii, exact coordinates or pixel errors.',
        decisions=dict(accept='Meaning is preserved; reason is empty.',
            **({'refine':'State the clear semantic defect in one short sentence.'} if can_refine else {}),
            unresolved='Evidence is insufficient, or a clear semantic defect remains with no refinement available.'),
        output_schema={'decision':'|'.join(choices),'reason':'One short sentence, empty for accept. No coordinates or replacement geometry.'})

def validate(answer,can_refine,allow_accept=True):
    if not isinstance(answer,dict):raise ValueError('Return a JSON object')
    allowed={'accept','unresolved'}|({'refine'} if can_refine else set())
    if not allow_accept:allowed.discard('accept')
    if answer.get('decision') not in allowed:raise ValueError('Choose one of '+','.join(sorted(allowed)))
    reason=answer.get('reason','')
    if not isinstance(reason,str) or (answer['decision']!='accept' and not reason.strip()):raise ValueError('Non-accept decisions require a short reason')
    return dict(decision=answer['decision'],reason=reason)

def recover(crop,row,support,out,model,config,locks,observation_path=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);initial=copy.deepcopy(row);current=copy.deepcopy(row)
    write(out/'before_geometry.json',initial);crop.save(out/'clean.png')
    observation=read(observation_path) if observation_path and Path(observation_path).exists() else None
    max_refine=config.get('block_shape_max_refinements',1);revisions=0;history=[];issues=[];accepted=False
    try:
        while True:
            stage=out/f'check_{revisions:02d}'
            with locks['cpu']:current_path=view(crop,current,stage/'input',observation)
            images=[('CLEAN',out/'clean.png'),('CURRENT fitted boundary and observed samples',current_path)]
            write(stage/'current_geometry.json',current)
            q=prompt(revisions<max_refine)
            complete=current.get('shape_observation_status','ok')=='ok'
            if not complete:q['known_incomplete']='CURRENT was reported partial. Choose refine or unresolved.'
            answer=model.ask(stage/'request',q,images,
                lambda v:validate(v,revisions<max_refine,complete))
            write(stage/'answer.json',answer);history.append(dict(check=revisions,**answer))
            if answer['decision']=='accept':accepted=True;break
            if answer['decision']=='unresolved':
                issues.append(dict(stage='block_shape_review',target=row['id'],state='unresolved',reason=answer['reason'],retained='current_geometry'));break
            revisions+=1
            generation=out/f'revision_{revisions:02d}'/'generation'
            revised=block_shapes.recover(crop,initial,support,generation,model,feedback=[answer['reason']])
            current=revised;observation=read(generation/'answer.json')
    except Exception as exc:
        issue=dict(stage='block_shape_review',target=row['id'],state='failed',error=str(exc),retained='last_valid_geometry')
        issues.append(issue);write(out/'error.json',issue)
    summary=dict(protocol=VERSION,accepted_by_model=accepted,refinements=revisions,max_refinements=max_refine,
        selected_source=current['coordinate_source'],history=history,issues=issues,
        visual_accuracy_measured=False)
    current['shape_geometry_review']=dict(directory=str(out.resolve()),accepted_by_model=accepted,refinements=revisions,selected_source=summary['selected_source'])
    write(out/'after_geometry.json',current);write(out/'summary.json',summary)
    return current,issues
