"""Measure and export saved recognized text with no model requests."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha
from shape_integration.texts import layout,colors,export,options,PROTOCOL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--recognition',type=Path,required=True)
    p.add_argument('--original-recognition',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    a.output.mkdir(parents=True)
    data=read(a.recognition);rows=deepcopy(data['texts'])
    if not rows:raise ValueError('Empty recognition cannot produce a valid text reconstruction')
    if a.original_recognition:
        original={r['id']:r for r in read(a.original_recognition)['texts']}
        for row in rows:
            if row['id'] in original:row['geometry_prior_bbox']=original[row['id']]['bbox_2d']
            if row.get('parent_id') in original:
                row['parent_geometry_prior_bbox']=original[row['parent_id']]['bbox_2d']
    before=sha(a.recognition)
    save(a.output/'provenance.json',{'protocol':PROTOCOL,'api_calls':0,'recognition':str(a.recognition.resolve()),
         'recognition_sha256':before,'image':str(a.image.resolve()),'image_sha256':sha(a.image),
         'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'src').rglob('*.py')}})
    opt=options(read(ROOT/'config.json'));slots=threading.BoundedSemaphore(1)
    fitted,errors=layout.fit_all(rows,a.output/'typography',slots,opt,source=a.image)
    fitted,audit=colors.recover(fitted,a.image,a.output/'color_assets')
    save(a.output/'color_recovery.json',audit)
    errors=[*data.get('errors',[]),*errors]
    with Image.open(a.image) as im:size=list(im.size)
    save(a.output/'scene.json',{'protocol':PROTOCOL,'source_size':size,'texts':fitted,'errors':errors})
    counts=export.final(fitted,a.image,a.output,slots,opt)
    save(a.output/'summary.json',{'state':'completed_with_unresolved' if errors else 'completed',
         'counts':counts,'errors':errors,'api_calls':0,'visual_accuracy_verified':False})
    assert sha(a.recognition)==before
    print(a.output,flush=True)


if __name__=='__main__':main()
