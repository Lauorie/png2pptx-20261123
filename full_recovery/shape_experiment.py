"""Export paired shape experiment evidence after both fresh workflows finish."""
import json
from pathlib import Path
import sys


def compare(control, improved):
    import fitz
    from PIL import Image
    read=lambda p:json.loads(p.read_text())
    a=read(control/'source_manifest.json')['images'];b=read(improved/'source_manifest.json')['images']
    if a!=b:raise ValueError('The experiment arms must use identical original inputs')
    output=control/'delivery';document=fitz.open();cases=[]
    for item in a:
        ident=item['id'];images=[Path(item['original']),control/'delivery'/ident/'preview.png',
                               improved/'delivery'/ident/'preview.png']
        with Image.open(images[0]) as image:w,h=image.size
        height=320*h/w;page=document.new_page(width=960,height=height+48)
        for column,(label,path) in enumerate(zip(('original','control','color + SAM radius'),images)):
            page.insert_text((column*320+8,17),ident+' '+label)
            if path.exists():page.insert_image(fitz.Rect(column*320,30,(column+1)*320,30+height),filename=str(path))
        audits=[]
        for path in (improved/'geometry/cases'/ident).rglob('radius_refinement.json'):
            value=read(path)
            audits.extend(value if isinstance(value,list) else [value])
        cases.append({'id':ident,'radius_attempts':len(audits),
                      'radius_applied':sum(bool(r.get('applied')) for r in audits),'radius_audits':audits})
    document.save(output/'original_control_improved_27.pdf');document.close()
    result={'control':str(control),'improved':str(improved),'samples':len(a),
            'fresh_independent_predictions_in_both_arms':True,'cases':cases,
            'control_summary':read(control/'summary.json'),'improved_summary':read(improved/'summary.json'),
            'visual_accuracy_verified':False}
    (output/'paired_experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':compare(*(Path(p) for p in sys.argv[1:]))
