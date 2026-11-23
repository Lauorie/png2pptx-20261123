"""Measure crop-margin sensitivity without changing the reconstructed text."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shape_integration.common import read,save,sha
from shape_integration.texts.foreground import contrast_ink
from shape_integration.texts.render_feedback import window,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--texts',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--margins',type=int,nargs='+',default=[4,2,0])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if any(p<0 for p in args.margins):raise ValueError('Do not crop inside predicted text bbox')
    args.output.mkdir(parents=True,exist_ok=False)
    source=Image.open(args.image).convert('RGB');render=Image.open(args.texts/'native.png').convert('RGB')
    if source.size!=render.size:raise ValueError('Source/render size mismatch')
    row=next(r for r in read(args.texts/'scene.json')['texts'] if r['id']==args.id)
    hashes={str(p):sha(p) for p in [args.image,args.texts/'native.png',args.texts/'scene.json']}
    results=[];tiles=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for margin in args.margins:
        roi=window([row],source.size,padding=margin)
        original=source.crop(roi);prediction=render.crop(roi)
        src=np.asarray(original,dtype=np.float32);rgb=np.asarray(prediction,dtype=np.float32)
        core=[0,0,src.shape[1],src.shape[0]]
        ref=contrast_ink(src,core);pred=contrast_ink(rgb,core)
        dice=float(2*(ref*pred).sum()/max(float((ref*ref).sum()+(pred*pred).sum()),1e-9))
        overlap=np.minimum(ref,pred);distance=np.linalg.norm(src-rgb,axis=2)/(255*math.sqrt(3))
        color=float(((1-distance)*overlap).sum()/max(float(overlap.sum()),1e-9))
        score=dice*(.75+.25*color)
        if margin==4:
            assert abs(score-evidence(source,render,render,[row])['after']['score'])<1e-6
        record={'margin_px':margin,'roi':roi,'size':list(original.size),'score':score,
                'mask_dice':dice,'color_agreement':color,'source_mass':float(ref.sum()),'render_mass':float(pred.sum())}
        results.append(record)
        original.save(args.output/f'margin_{margin}_source.png');prediction.save(args.output/f'margin_{margin}_render.png')
        # Uniform scale for every crop, no per-panel size normalization.
        scale=6;w,h=original.size
        tile=Image.new('RGB',(2*w*scale+40,h*scale+70),'white');draw=ImageDraw.Draw(tile)
        draw.text((8,4),f'Margin {margin}px | crop {w}x{h} | CPU {score*100:.2f}',font=font,fill='black')
        draw.text((8,30),'Source',font=font,fill='black');draw.text((w*scale+24,30),'Unchanged PPT text',font=font,fill='black')
        tile.paste(original.resize((w*scale,h*scale),Image.Resampling.NEAREST),(8,60))
        tile.paste(prediction.resize((w*scale,h*scale),Image.Resampling.NEAREST),(w*scale+24,60))
        tiles.append(tile)
    board=Image.new('RGB',(max(t.width for t in tiles),sum(t.height for t in tiles)),'white');y=0
    for tile in tiles:board.paste(tile,(0,y));y+=tile.height
    board.save(args.output/'comparison.png')
    save(args.output/'scores.json',{'id':row['id'],'text':row['text'],'original_bbox':row['bbox_2d'],
        'sources':hashes,'text_render_unchanged':True,'score_comparability':'Different ROIs; increases alone do not establish reconstruction improvement',
        'results':results})
    assert all(sha(Path(p))==h for p,h in hashes.items())
    for r in results:print(r,flush=True)


if __name__=='__main__':main()
