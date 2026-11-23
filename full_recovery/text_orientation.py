"""Readable orientation evidence for the existing locked-box attribute call."""
from copy import deepcopy
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps


def quarter_turn_ids(prompt):
    if prompt.get('stage')!='2_attributes_at_locked_boxes':return set()
    return {r['id'] for r in prompt.get('targets',[]) if r.get('kind','text')!='formula'
            and (b:=r.get('locked_bbox_2d')) and
            (abs(float(r.get('rotation_deg',0)))%180==90 or b[2]-b[0]<.55*(b[3]-b[1]))}


def augment(root,prompt,images):
    ids=quarter_turn_ids(prompt)
    targets=[r for r in prompt.get('targets',[]) if r['id'] in ids]
    if not targets or not images:return prompt,images
    with Image.open(images[0][1]) as opened:source=opened.convert('RGB')
    sheet=Image.new('RGB',(1800,220*len(targets)),'#edf0f3');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(targets):
        x0,y0,x1,y1=row['locked_bbox_2d'];w=x1-x0;h=y1-y0
        # Include context around the locked box: a wrong short box must not
        # hide the tail of a word and turn an attribute review into a guess.
        px=max(3,w*.2);py=max(3,h*.3)
        crop=source.crop((max(0,int(x0-px)),max(0,int(y0-py)),
                          min(source.width,int(x1+px+1)),min(source.height,int(y1+py+1))))
        for j,angle in enumerate((90,-90,0)):
            tile=ImageOps.contain(crop.rotate(angle,expand=True),(580,174),Image.Resampling.LANCZOS)
            draw.text((j*600+10,i*220+10),f'{row["id"]}: candidate {"ABC"[j]}',fill='black')
            sheet.paste(tile,(j*600+(600-tile.width)//2,i*220+36+(174-tile.height)//2))
    path=Path(root)/'orientation_candidates.png';path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
    result=deepcopy(prompt)
    result['rules']=[r for r in result['rules'] if not r.startswith('ROTATION IS NOT LOCKED.')]
    for row in result['targets']:
        if row['id'] in ids:row.pop('rotation_deg',None)
    if len(ids)==len(result['targets']):
        result['rules']=[
            '每个ID只返回一条文字记录。文字框已锁定，不改框，不合并，不拆分。',
            '读取对照图中同一ID的完整文字、字体、颜色和样式；runs保留上下标。不要输出字号或背景。',
            '锁定框没有包含完整文字，或归属不清楚时，返回ambiguous和空units；不要猜测。']
        result['response_format']['items'][0]['units'][0].pop('rotation_deg',None)
    result['rules'].append('方向只看该ID的A/B/C对照图，选择所有字母本身正立的一张：字母的上方应朝上，不能横倒或倒立。若字母正立且自上而下逐字排列，layout_mode为stacked_vertical；正常一行单词为single_line。只返回orientation_candidate和文字属性，不返回角度；无法区别则返回ambiguous。')
    result['response_format']['items'][0]['units'][0]['orientation_candidate']='A/B/C for IDs on the orientation board'
    # The board already contains every target's complete source pixels. Extra
    # unrotated views anchor the model to the source orientation instead of the
    # upright glyph choice. Keep context only for targets absent from the board.
    return result,[('ORIENTATION: choose the upright readable candidate for each ID.',path),
                   *(images if len(ids)!=len(result['targets']) else [])]


def bind_validator(validator,ids):
    """Visual choice is the sole angle authority; caches retain its proof."""
    def validate(raw):
        value=deepcopy(raw);choices=value.get('_orientation_candidates',{}).copy()
        for item in value.get('items',[]):
            if item.get('id') not in ids:continue
            if item.get('status')=='resolved':
                units=item.get('units',[])
                if len(units)!=1:raise ValueError('One orientation choice per locked text ID')
                unit=units[0];choice=unit.get('orientation_candidate')
                if choice not in ('A','B','C'):raise ValueError('Choose orientation_candidate A, B or C, or ambiguous')
                unit['rotation_deg']={'A':90,'B':-90,'C':0}[choice];choices[item['id']]=choice
                # A/B selects a rotated readable baseline. Upright stacked
                # glyphs are a different representation (rotation zero), so
                # it cannot also be the layout of this selected candidate.
                count=unit.get('visual_line_count',1)
                if isinstance(count,int) and not isinstance(count,bool) and count>=1 and not (choice=='C' and unit.get('layout_mode')=='stacked_vertical'):
                    unit['layout_mode']='single_line' if count==1 else 'multiline'
            elif item.get('status')=='located':
                choice=choices.get(item['id'])
                if choice not in ('A','B','C'):raise ValueError('Normalized orientation requires its candidate proof')
                item['rotation_deg']={'A':90,'B':-90,'C':0}[choice]
        result=validator(value);result['_orientation_candidates']=choices;return result
    return validate
