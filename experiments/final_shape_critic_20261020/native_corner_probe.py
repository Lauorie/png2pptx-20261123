"""Minimal six-point -> native DrawingML corner -> real PPT render experiment."""
from pathlib import Path
import argparse
import copy
import json
import math
import os
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from lxml import etree
from shape_integration.common import read,save,compare,sha
from shape_integration.fitting.adapter import fit_object,transform_fit
from shape_integration.fitting.contracts import observation_schema,validate_observation
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.scene import native_parameters,write_pptx
from shape_integration.styling.export import render_native
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.util import Pt


def observation(points):
    ids=[f'P{i}' for i in range(len(points))]
    return {'object_id':'L','status':'ok','reason':'Six observed contour vertices', 'uncertain':[],
        'points':[{'id':pid,'xy':list(q)} for pid,q in zip(ids,points)],
        'parts':[{'id':'outline','kind':'path','closed':True,
            'segments':[{'kind':'line','point_ids':[pid,ids[(i+1)%len(ids)]]} for i,pid in enumerate(ids)]}]}


def point_scene(raw,size):
    frame=FitFrame(tuple(size),tuple(size),tuple(size),(0,0,*size))
    fit=transform_fit(fit_object(raw,frame,object_id='L',namespace='native_corner_probe'))
    if fit['fit_status']!='ok':raise ValueError('Point fitting failed')
    parts={}
    for i,p in enumerate(fit['parts']):
        p['native_parameters_original']=native_parameters(p,fit['frame']);parts[str(i)]=p
    scene={'source_size':size,'objects':{'L':{'id':'L','state':'unreviewed_probe','part_keys':list(parts)}},'parts':parts}
    return scene,fit


def export(scene,folder,filled):
    folder.mkdir(parents=True,exist_ok=True);ppt=folder/'native_corner.pptx';w=scene['source_size'][0]
    def paint(shape,part,scale):
        if filled:
            shape.fill.solid();shape.fill.fore_color.rgb=RGBColor(40,110,210);shape.line.fill.background()
        else:
            shape.fill.background();shape.line.color.rgb=RGBColor(210,167,109)
            shape.line.width=Pt(1.2*13.333333*72/w);shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    counts=write_pptx(scene,ppt,style_writer=paint)
    with zipfile.ZipFile(ppt) as z:
        xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        presets=xml.xpath('//a:prstGeom/@prst',namespaces=ns)
        images=[n for n in z.namelist() if n.startswith('ppt/media/')]
    native=render_native(ppt,folder/'render',scene['source_size'])
    return native,{'counts':counts,'preset_xml':presets,'embedded_images':len(images),
        'native_corner':presets==['corner'] and not images,'pptx':str(ppt)}


def main():
    from api_compare import api_runtime
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    worker=read(Path(os.environ['PPT_API_WORK_ROOT'])/'settings.json');project=Path(worker['project_root'])
    synthetic=[]
    for fh,fv in ((False,False),(True,False),(False,True),(True,True)):
        points=np.array([[20,20],[380,20],[380,140],[460,140],[460,200],[20,200]],float)
        if fh:points[:,0]=480-points[:,0]
        if fv:points[:,1]=220-points[:,1]
        folder=run/'synthetic'/f'h{int(fh)}_v{int(fv)}';scene,fit=point_scene(observation(points),[500,240])
        native,result=export(scene,folder,True)
        source=Image.new('RGB',(500,240),'white');ImageDraw.Draw(source).polygon([tuple(q) for q in points],fill=(40,110,210));source.save(folder/'source.png')
        truth=Image.new('L',(500,240),0);ImageDraw.Draw(truth).polygon([tuple(q) for q in points],fill=255)
        actual=np.asarray(Image.open(native).convert('RGB'));mask=(actual[:,:,0]<100)&(actual[:,:,2]>150);target=np.asarray(truth)>0
        result.update(flip_h=fh,flip_v=fv,iou=float(np.sum(mask&target)/np.sum(mask|target)))
        if not result['native_corner'] or result['iou']<.98:raise AssertionError('Native corner render did not preserve six points')
        save(folder/'fit.json',fit);save(folder/'result.json',result)
        compare([folder/'source.png',native],['Six observed vertices','Actual native corner PPT'],folder/'comparison.png',width_cap=1500)
        synthetic.append(result);print('NATIVE_CORNER',fh,fv,result['iou'],flush=True)
    # This is a source-only object probe, not reuse of the old predicted contour.
    base=project/'runs/pptbench_full_continue_llmshare_20261019_v8'
    items={i['id']:i for i in read(base/'source_manifest.json')['images']};item=items['task_0897']
    prior=read(base/'shape/geometry/cases/task_0897/fitting/scene.json')
    boxes=[prior['objects'][oid]['candidate']['bbox_2d'] for oid in ['G008','F1_N001']]
    b=[math.floor(min(q[0] for q in boxes))-12,math.floor(min(q[1] for q in boxes))-12,
       math.ceil(max(q[2] for q in boxes))+12,math.ceil(max(q[3] for q in boxes))+12]
    folder=run/'task_0897';folder.mkdir(exist_ok=True)
    with Image.open(item['original']) as im:
        b=[max(0,b[0]),max(0,b[1]),min(im.width,b[2]),min(im.height,b[3])]
        crop=im.convert('RGB').crop(b);crop.save(folder/'source_crop.png')
    runtime=api_runtime(project)
    prompt={'task':'Observe only the orange/tan dashed outer grouping boundary in this CLEAN crop. '
        'Trace its complete connected contour, including any inward step; ignore inner text boxes, '
        'red borders and arrows. Determine from the source whether it is one connected outline or '
        'separate shapes. A standard orthogonal L has six turns; do not force that topology if unsupported. '
        'Locate stroke-centerline vertices. Corresponding coordinates of a visibly horizontal or '
        'vertical source edge should be shared. Return one closed line-segment path when supported, '
        'otherwise unsupported with the source reason.',
        'coordinates':'Pixel centers in this crop only, not full image coordinates.',
        'output_schema':observation_schema(crop.size),'object_id':'L'}
    raw=runtime.call(folder/'calls','source_points','Observe the supplied source contour. Return JSON only.',
        prompt,[('CLEAN source only',folder/'source_crop.png')],
        lambda r:validate_observation(r,'L',crop.size),32768)
    save(folder/'observation.json',raw)
    real={'state':raw['status'],'source_crop_original':b,'original_source_sha256':sha(item['original']),
        'source_only_new_api_observation':True,'prior_predictions_used_for_crop_only':True,
        'style_scope':'Fixed illustrative dashed stroke, geometry validation only.'}
    if raw['status']=='ok':
        scene,fit=point_scene(raw,list(crop.size));native,result=export(scene,folder/'output',False)
        real.update(result,points=len(raw['points']),parts=len(raw['parts']))
        overlay=crop.copy();draw=ImageDraw.Draw(overlay);table={p['id']:p['xy'] for p in raw['points']}
        for p in raw['parts']:
            for segment in p.get('segments',[]):draw.line([tuple(table[k]) for k in segment['point_ids']],fill='#C000E0',width=2)
        for pid,xy in table.items():draw.ellipse((xy[0]-3,xy[1]-3,xy[0]+3,xy[1]+3),fill='red');draw.text((xy[0]+4,xy[1]+2),pid,fill='black')
        overlay.save(folder/'observed_points.png');save(folder/'fit.json',fit)
        compare([folder/'source_crop.png',folder/'observed_points.png',native],['CLEAN crop','Fresh API observed vertices','Native PPT outline only'],folder/'comparison.png',width_cap=2100)
    save(folder/'result.json',real)
    save(run/'summary.json',{'synthetic':synthetic,'real_source':real,'full_batch_modified':False,
        'source_geometry_manual_review_required':True})
    print('REAL_SOURCE_RESULT',json.dumps(real),flush=True)


if __name__=='__main__':main()
