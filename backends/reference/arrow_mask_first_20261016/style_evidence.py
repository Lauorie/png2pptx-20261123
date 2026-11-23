"""Small source/render detail pairs and optional measurements, never style gates."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from lib import style,fit
import style_contract
from transport.common import write_json as write


def measurements(crop,rows,supports,out):
    result=[]
    for row in rows:
        mask=supports.get(row['id'])
        measured=style_contract.provisional(crop,mask,row)
        evidence=measured.get('evidence',{})
        result.append(dict(id=row['id'],source='optional final-geometry pixel estimate',
            mask_available=mask is not None,color=measured.get('color'),width_px=measured.get('width_px'),
            dash=measured.get('dash'),review_reasons=measured.get('review_reasons',[]),
            geometry_offset_px=evidence.get('median_geometry_offset_px'),state=measured.get('state')))
    write(Path(out)/'measurements.json',result)
    return result


def details(crop,rows,render_path,out):
    """Same boxes and scale on both sides. Labels attach to physical path IDs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rendered=Image.open(render_path).convert('RGB')
    if rendered.size!=crop.size:raise ValueError('Style detail source/render grids differ')
    tiles=[];records=[]
    for row in rows:
        if row['kind']=='block_arrow':continue
        s,q,_=style.sample(row,step=1)
        middle=q[np.argmin(abs(s-s[-1]*.5))]
        width=row['style'].get('width_px') or row.get('width_estimate') or 2
        # View extent only: no width/type or geometry is inferred by this window.
        radius=max(24.,width*6)
        for label,p in [('start',q[0]),('shaft',middle),('end',q[-1])]:
            lo=np.maximum(0,np.floor(p-radius)).astype(int)
            hi=np.minimum(crop.size,np.ceil(p+radius+1)).astype(int)
            box=[*lo.tolist(),*hi.tolist()]
            if hi[0]<=lo[0] or hi[1]<=lo[1]:continue
            pair=Image.new('RGB',(400,220),'white');draw=ImageDraw.Draw(pair)
            draw.text((4,4),row['id']+' / '+label,fill='black')
            draw.text((4,20),'SOURCE',fill='black');draw.text((204,20),'PPT',fill='black')
            for x,im in [(0,crop),(200,rendered)]:
                part=im.crop(box);part.thumbnail((196,180),Image.Resampling.LANCZOS)
                # Resize both equally to the same readable detail size.
                ratio=min(196/(hi[0]-lo[0]),180/(hi[1]-lo[1]))
                part=im.crop(box).resize((round((hi[0]-lo[0])*ratio),round((hi[1]-lo[1])*ratio)),Image.Resampling.NEAREST)
                pair.paste(part,(x+2,38))
            tiles.append(pair);records.append(dict(id=row['id'],endpoint=label,box_local=box))
    pages=[]
    for i in range(0,len(tiles),6):
        selected=tiles[i:i+6];page=Image.new('RGB',(800,220*((len(selected)+1)//2)),'white')
        for j,tile in enumerate(selected):page.paste(tile,((j%2)*400,(j//2)*220))
        path=out/f'page_{i//6:02d}.png';page.save(path);pages.append(path)
    write(out/'windows.json',records)
    return [('SOURCE/PPT paired details, identical window and scale',p) for p in pages]
