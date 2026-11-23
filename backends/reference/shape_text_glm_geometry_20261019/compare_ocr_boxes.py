"""Independent CPU OCR on paired crops; do not alter text or source pixels."""
import argparse
import json
import os
from pathlib import Path
import time
import unicodedata


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def overlap(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return {'iou':intersection/max(area(a)+area(b)-intersection,1e-9),
            'coverage_a':intersection/max(area(a),1e-9),'coverage_b':intersection/max(area(b),1e-9)}


def normalized(text):
    return ''.join(unicodedata.normalize('NFKC',text).casefold().split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crop-dir',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    args.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    from paddleocr import PaddleOCR
    print('Loading local PaddleOCR on CPU',flush=True)
    start=time.monotonic()
    ocr=PaddleOCR(ocr_version='PP-OCRv5',
        text_detection_model_name='PP-OCRv5_server_det',
        text_detection_model_dir=str(args.models/'PP-OCRv5_server_det'),
        text_recognition_model_name='PP-OCRv5_server_rec',
        text_recognition_model_dir=str(args.models/'PP-OCRv5_server_rec'),
        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
        text_det_limit_side_len=2048,text_det_limit_type='max',device='cpu',cpu_threads=2,
        enable_hpi=False,enable_mkldnn=False)
    frame=json.loads((args.crop_dir/'frame.json').read_text())
    row=next(r for r in json.loads(args.scene.read_text())['texts'] if r['id']==args.id)
    map_box=lambda box:[(v-frame['box'][i%2]+.5)*frame['scale']-.5+frame['offset'][i%2] for i,v in enumerate(box)]
    unmap=lambda box:[(v+.5-frame['offset'][i%2])/frame['scale']+frame['box'][i%2]-.5 for i,v in enumerate(box)]
    current=map_box(row['bbox_2d']);records={};panels=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for name,filename in [('source','clean.png'),('ppt','render.png')]:
        begin=time.monotonic()
        predictions=list(ocr.predict(str(args.crop_dir/filename)))
        if len(predictions)!=1:raise ValueError('Expected one page')
        raw=predictions[0].json
        if isinstance(raw,str):raw=json.loads(raw)
        write(args.output/(name+'_raw.json'),raw)
        data=raw.get('res',raw)
        boxes=[]
        for text,confidence,poly in zip(data['rec_texts'],data['rec_scores'],data['rec_polys']):
            b=[min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)]
            boxes.append({'text':text,'confidence':float(confidence),'crop_bbox':b,'source_bbox':unmap(b),'polygon':poly})
        matches=[b for b in boxes if normalized(b['text'])==normalized(row['text'])]
        # Content alone is insufficient when a crop contains repeated labels.
        for b in matches:b['vs_saved_bbox']=overlap(b['crop_bbox'],current)
        records[name]={'elapsed_seconds':time.monotonic()-begin,'boxes':boxes,'matches':matches}
        canvas=Image.open(args.crop_dir/filename).convert('RGB');draw=ImageDraw.Draw(canvas)
        draw.rectangle(current,outline='#e00090',width=3)
        draw.text((max(0,current[0]),max(0,current[1]-30)),'Saved bbox',font=font,fill='#e00090')
        for b in matches:
            draw.line([tuple(q) for q in b['polygon']]+[tuple(b['polygon'][0])],fill='#008850',width=4)
            x,y=b['crop_bbox'][:2];draw.text((max(0,x),max(0,y-60)),f'OCR: {b["text"]}',font=font,fill='#008850')
        canvas.save(args.output/(name+'_overlay.png'));panels.append(canvas)
        print(name,json.dumps({'boxes':boxes,'match_count':len(matches)},ensure_ascii=False),flush=True)
    comparisons=[]
    for a in records['source']['matches']:
        for b in records['ppt']['matches']:
            comparisons.append({'source_ocr':a,'ppt_ocr':b,'overlap':overlap(a['crop_bbox'],b['crop_bbox'])})
    report={'target':{'id':row['id'],'text':row['text'],'source_bbox':row['bbox_2d'],'crop_bbox':current},
            'frame':frame,'model':'PP-OCRv5_server_det+PP-OCRv5_server_rec','device':'cpu',
            'results':records,'source_vs_ppt_ocr':comparisons,'total_seconds':time.monotonic()-start,
            'note':'OCR bbox overlap is geometry evidence, not pixel fidelity or font/color correctness. Multiple matches require disambiguation.'}
    write(args.output/'report.json',report)
    w,h=panels[0].size
    board=Image.new('RGB',(w*2,h+40),'white');draw=ImageDraw.Draw(board)
    for i,(name,panel) in enumerate(zip(['Original crop','PPT text render'],panels)):
        board.paste(panel,(i*w,40));draw.text((i*w+10,5),name,font=font,fill='black')
    board.save(args.output/'comparison.png')
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
