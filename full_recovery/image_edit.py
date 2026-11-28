"""Optional single edit for an owned decorative asset; never a diagram fallback.

The existing texture decision supplies the prompt. No new routing/critic loop.
Native geometry is removed only after a nonempty, aligned RGBA is validated.
"""
import base64
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_propagation, label

from .service_endpoints import IMAGE_EDIT_BASE_URL as DEFAULT_BASE_URL, image_edit_base


def transparent_exterior(image, tolerance=18, *, content_box=None, adaptive=True):
    """Remove border-connected white, preserving white interiors and source alpha."""
    rgba=np.asarray(image.convert('RGBA')).copy()
    # Only a white canvas is requested. Never guess transparency from a
    # dominant object color or remove all pixels of that color globally.
    whiteness=np.min(rgba[:,:,:3],axis=2)
    allowed=(whiteness>=255-tolerance)|(rgba[:,:,3]==0)
    # The editor can add faint neutral paper grain to the requested white
    # canvas. Estimate its floor from the padded border; do not key out a
    # dominant object colour or enclosed white regions.
    rgb=rgba[:,:,:3].astype(float)
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    if adaptive and np.median(border.min(1))>=237 and np.quantile(np.ptp(border,axis=1),.95)<=18:
        # White padded canvases sometimes have a darker neutral paper rim.
        # Its observed floor, not a fixed near-white cutoff, defines the key.
        # The border-white/chroma conditions above still exclude colored edges.
        floor=max(180,min(255-tolerance,float(np.quantile(border.min(1),.001))-12))
        allowed|=(whiteness>=floor)&(np.ptp(rgb,axis=2)<=18)
    seed=np.zeros(allowed.shape,bool)
    seed[0]=allowed[0];seed[-1]=allowed[-1];seed[:,0]=allowed[:,0];seed[:,-1]=allowed[:,-1]
    if content_box is not None:
        # The editor's synthetic white padding is known background too. A
        # generated dark paper rim can seal every canvas edge and otherwise
        # prevent the exterior flood from starting at all. Never seed inside
        # the source crop: enclosed white object interiors remain opaque.
        x0,y0,x1,y1=map(int,content_box)
        padding=np.ones(allowed.shape,bool);padding[y0:y1,x0:x1]=False
        seed|=allowed&padding
    outside=binary_propagation(seed,mask=allowed)
    rgba[outside,3]=0
    if not np.any(rgba[:,:,3]):raise ValueError('Edited asset is empty after background removal')
    return Image.fromarray(rgba,'RGBA')


def clear_padding_fragments(image, content_box):
    """Remove isolated neutral rim noise wholly outside the supplied crop.

    Components reaching the actual content remain intact, including genuinely
    clipped objects. This is specific to the editor's known added white margin.
    """
    rgba=np.asarray(image.convert('RGBA')).copy()
    components,_=label(rgba[:,:,3]>0,structure=np.ones((3,3),bool))
    edges=set(np.unique(np.r_[components[0],components[-1],components[:,0],components[:,-1]]))
    x0,y0,x1,y1=map(int,content_box)
    content=set(np.unique(components[y0:y1,x0:x1]));removed=0
    for ident in edges-content-{0}:
        mask=components==ident
        if np.max(np.ptp(rgba[mask,:3].astype(int),axis=1))<=18:
            removed+=int(mask.sum());rgba[mask,3]=0
    return Image.fromarray(rgba,'RGBA'),removed


def cleanup_mask(size,prompt):
    mask=Image.new('L',size,0);draw=ImageDraw.Draw(mask)
    for label in prompt.get('native_labels',[]):
        b=label['box']
        draw.rectangle((math.floor(b[0]-1),math.floor(b[1]-1),math.ceil(b[2]+1),math.ceil(b[3]+1)),fill=255)
    for route in prompt.get('removal_paths',[]):
        points=[tuple(p) for p in route['points']]
        if len(points)>1:draw.line(points,fill=255,width=max(1,math.ceil(route.get('width_px',1)+2)))
    if prompt.get('edit_box') is not None:
        restriction=Image.new('L',size,0)
        ImageDraw.Draw(restriction).rectangle(tuple(prompt['edit_box']),fill=255)
        mask=Image.fromarray(np.minimum(np.asarray(mask),np.asarray(restriction)))
    return mask


# A pixel is lettering when it is this far, in RGB, from the colour around its label,
# and the same lettering when it is this near to the label's own colour.
LABEL_INK_DISTANCE = 60.
# Share of a label's source lettering still present in the edited picture that means the label was not removed.
LABEL_LEFT_SHARE = .5


def clear_labels_left(source, patch, prompt):
    """Remove from the edited picture every label the editor returned untouched.

    The slide adds these labels as text of its own. The editor is asked to take
    them out of the picture and sometimes leaves them, so the text ends up on
    top of its own copy. Whether a label is still there is measured, not asked:
    its lettering in the source must reappear, in its own colour and place,
    after editing.

    Args:
        source: The source crop, RGBA.
        patch: The edited picture aligned to the crop, RGBA.
        prompt: The edit request; ``native_labels`` holds each label's box in crop pixels.

    Returns:
        The patch with those labels painted out from the source surroundings, and how many there were.
    """
    import cv2
    before = np.asarray(source.convert('RGB')).astype(float); after = np.asarray(patch.convert('RGB')).astype(float)
    height, width = before.shape[:2]
    stale = np.zeros((height, width), 'uint8'); count = 0
    for label in prompt.get('native_labels', []):
        b = label['box']
        x0, y0 = max(0, math.floor(b[0]-1)), max(0, math.floor(b[1]-1))
        x1, y1 = min(width, math.ceil(b[2]+1)+1), min(height, math.ceil(b[3]+1)+1)
        if x1 <= x0 or y1 <= y0:
            continue
        ground = np.median(before[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        ink = np.linalg.norm(before[y0:y1, x0:x1]-ground, axis=2) > LABEL_INK_DISTANCE
        if not ink.any():
            continue
        # The same lettering, not merely something drawn there: an editor that
        # fills the place with the body's own colour has removed the label.
        colour = np.median(before[y0:y1, x0:x1][ink], axis=0)
        kept = (np.linalg.norm(after[y0:y1, x0:x1]-colour, axis=2) < LABEL_INK_DISTANCE).astype('uint8')
        kept = cv2.dilate(kept, np.ones((3, 3), 'uint8')) > 0
        if np.mean(kept[ink]) >= LABEL_LEFT_SHARE:
            stale[y0:y1, x0:x1] = 255; count += 1
    if not count:
        return patch, 0
    painted = cv2.inpaint(np.asarray(source.convert('RGB')), stale, 3., cv2.INPAINT_TELEA)
    result = np.asarray(patch.convert('RGBA')).copy()
    result[stale > 0, :3] = painted[stale > 0]; result[stale > 0, 3] = 255
    return Image.fromarray(result, 'RGBA'), count


def source_locked_icon(crop, edited, placement, prompt):
    """Apply cleanup only to explicit labels/overlays, in original pixels.

    Registered illustrations are source assets, not requests to redraw faces,
    nodes or hidden edges. Editor resolution and inferred silhouettes cannot
    change their geometry or original paint outside the removal masks.
    """
    source = crop.convert('RGBA');mask=cleanup_mask(source.size,prompt)
    if edited is None:
        import cv2
        patch = Image.fromarray(cv2.inpaint(np.asarray(source.convert('RGB')), np.asarray(mask), 3., cv2.INPAINT_TELEA)).convert('RGBA')
    else:
        patch = Image.new('RGBA', source.size, 'white')
        x, y, w, h = placement
        image = edited.convert('RGBA').resize((max(1, round(w)), max(1, round(h))), Image.Resampling.LANCZOS)
        patch.alpha_composite(image, (round(x), round(y)))
    left = 0
    if edited is not None:
        patch, left = clear_labels_left(source, patch, prompt)
    result = Image.composite(patch, source, mask)
    # This is an original-image crop, not a generated canvas. Whole-image
    # colour keying after a local edit can flood through a cleaned boundary
    # and erase grey fills outside the mask. Preserve RGBA outside it exactly;
    # downstream ownership clipping alone controls the asset's spatial scope.
    return result, {'source_pixels_policy': 'explicit_cleanup_masks.1',
                    'source_pixels_preserved': True, 'edited_fraction': float(np.mean(np.asarray(mask) > 0)),
                    'rgba_outside_cleanup_preserved': True, 'cleanup_fallback': edited is None,
                    'labels_left_by_editor': left}


class Editor:
    def __init__(self, config):
        self.config={'base_url': DEFAULT_BASE_URL, **config}
        self.config['base_url']=image_edit_base(self.config['base_url'])
        self.slots=threading.BoundedSemaphore(config.get('concurrency',2))

    def edit(self, crop, prompt, output, *, role='background'):
        import requests
        config=self.config;output=Path(output)
        session=requests.Session();session.trust_env=False
        model=config.get('model','SenseNova-U1.5-8B-MoT')
        url=config['base_url'].rstrip('/')
        side=config.get('image_size',1024)
        factor=(side-64)/max(crop.size)
        size=tuple(max(1,round(v*factor)) for v in crop.size)
        offset=((side-size[0])//2,(side-size[1])//2)
        canvas=Image.new('RGB',(side,side),'white')
        canvas.paste(crop.convert('RGB').resize(size,getattr(Image,'Resampling',Image).LANCZOS),offset)
        buf=BytesIO();canvas.save(buf,format='PNG');payload=buf.getvalue()
        if role not in ('background','texture','icon'):raise ValueError('Unknown image role')
        ownership=('Remove all foreground nodes, connectors and labels; reconstruct only the background behind them.'
            if role=='background' else
            'Restore only the named surface pattern. Remove separately drawn diagram nodes, connectors and labels; do not infer a network inside the texture.'
            if role=='texture' else
            'Preserve only the approved complex geometry, intrinsic dots, symbols and texture. '
            'Separately owned native labels, frames, nodes and obvious arrows are external content, '
            'not parts to restore in this image. Follow the explicit removal scope; do not redraw them.')
        if isinstance(prompt,dict):
            # The planner describes the target; ownership and coordinate
            # conversion have one author here, after the canvas is prepared.
            def canvas_box(bounds):
                return [round(offset[i%2]+bounds[i]*size[i%2]/crop.size[i%2],2) for i in range(4)]
            instructions=['清理给定截图，保留原有画法，不重新设计或美化。',
                '保留的非文字目标：'+prompt['target'],
                '移除的外部覆盖物：'+json.dumps(prompt.get('remove_overlays',[]),ensure_ascii=False),
                # Transparency is derived by the program from a white canvas; a drawn
                # checkerboard reaches the canvas edge and the asset is rejected.
                '背景由程序处理：目标以外必须是纯白(#FFFFFF)画布；忽略目标描述中关于透明背景的要求，不要画透明棋盘格。']
            labels=[{'text':r['text'],'canvas_box':canvas_box(r['box'])}
                    for r in prompt.get('native_labels',[])]
            if labels:instructions.append('以下文字由PPT另行提供，只从图片中移除这些文字；目标描述中的这些文字仅用于定位，不要求保留：'+json.dumps(labels,ensure_ascii=False))
            if prompt.get('edit_box') is not None:
                instructions.append('仅编辑当前输入画布像素框 '+str(canvas_box(prompt['edit_box']))+' 内的内容，框外保留原像素。')
            if prompt.get('native_frames'):
                instructions.append('已有原生外框仅作定位参照，不重画或移动；程序仅使用指定内部区域。')
            instruction='\n'.join(instructions)
        else:
            instruction=prompt.strip()  # Compatibility for direct callers.
        instruction+=('\nKeep the input canvas size, object position, silhouette, colors and perspective. '
            +ownership+' Put only the named target on a pure white background. '
            'Do not invent or duplicate diagram content, labels, shadows or transparency checkerboards.')
        identity=hashlib.sha256(payload+json.dumps({'config':config,'instruction':instruction},sort_keys=True).encode()).hexdigest()
        receipt_path=output.with_suffix('.json')
        if output.is_file() and receipt_path.is_file():
            previous=json.loads(receipt_path.read_text())
            if previous.get('identity')!=identity or previous.get('sha256')!=hashlib.sha256(output.read_bytes()).hexdigest():
                raise ValueError('Edited asset cache identity changed; use a new output directory')
            return previous
        with self.slots:
            response=session.post(url+'/images/edits',json={'model':model,'prompt':instruction,
                'images':[{'image_url':'data:image/png;base64,'+base64.b64encode(payload).decode()}],
                'size':f'{side}x{side}','output_format':'png','response_format':'b64_json',
                'img_config':{'width':side,'height':side,'dynamic_resolution':False,'seed':0}},
                timeout=(10,config.get('timeout_seconds',180)))
        response.raise_for_status();data=response.json()
        records=data.get('data',[])
        if not records:
            message=data.get('choices',[{}])[0].get('message',{})
            records=message.get('images',[]) or message.get('content',[])
            if not isinstance(records,list):raise ValueError('Editor did not return an image')
        encoded=None
        for row in records:
            if not isinstance(row,dict):continue
            encoded=row.get('b64_json')
            value=row.get('image_url',row.get('url',''))
            value=value.get('url','') if isinstance(value,dict) else value
            if not encoded and isinstance(value,str) and value.startswith('data:image/'):
                encoded=value.split(',',1)[1]
            if encoded:break
        if not encoded:raise ValueError('Editor response needs inline image bytes; arbitrary remote URLs are not fetched')
        if encoded.startswith('data:image/'):encoded=encoded.split(',',1)[1]
        encoded=''.join(encoded.split())
        with Image.open(BytesIO(base64.b64decode(encoded,validate=True))) as opened:
            if opened.width*opened.height>16777216:raise ValueError('Edited image is too large')
            if opened.width!=opened.height:
                raise ValueError('Editor changed canvas aspect ratio')
            # Undo only the known padding/resizing transform. Never stretch an
            # edited silhouette to a new bounding box inferred from its output.
            aligned=opened.convert('RGBA').resize((side,side),getattr(Image,'Resampling',Image).LANCZOS)
            output.parent.mkdir(parents=True,exist_ok=True)
            aligned.save(output.with_name(output.stem+'_raw_canvas.png'))
            content_box=[*offset,offset[0]+size[0],offset[1]+size[1]]
            result=transparent_exterior(aligned,content_box=content_box)
            result,padding_removed=clear_padding_fragments(result,content_box)
        bounds=result.getchannel('A').getbbox()
        if bounds is None:raise ValueError('Edited asset is empty after padding cleanup')
        if bounds[0]==0 or bounds[1]==0 or bounds[2]==side or bounds[3]==side:
            output.parent.mkdir(parents=True,exist_ok=True)
            aligned.save(output.with_name(output.stem+'_rejected_canvas.png'))
            raise ValueError('Edited object reaches padded canvas edge; preserve native geometry')
        # Keep the complete edited silhouette, including any repaired edge
        # beyond the original crop. Placement uses the known letterbox transform.
        bounds=(max(0,bounds[0]-1),max(0,bounds[1]-1),min(side,bounds[2]+1),min(side,bounds[3]+1))
        result=result.crop(bounds)
        alpha=np.asarray(result.getchannel('A'))
        if np.mean(alpha>0)<.01:raise ValueError('Edited foreground is too small')
        output.parent.mkdir(parents=True,exist_ok=True);result.save(output)
        receipt={'model':model,'base_url':url,'protocol':'sensenova_json_images/edits','role':role,
                'prompt':instruction,'canvas_size':list(crop.size),'alpha_method':'border_connected_white',
                'foreground_fraction':float(np.mean(alpha>0)),'requests':1,'identity':identity,
                'padding_fragment_pixels_removed':padding_removed,
                'source_offset_xy':[(bounds[i]-offset[i])/factor for i in (0,1)],
                'source_extent_wh':[(bounds[i+2]-bounds[i])/factor for i in (0,1)],
                'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
        receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        return receipt


def replace_objects(base, output, assets):
    """Compatibility entry; all image placement uses the same owner commit."""
    from .image_assets import commit
    rows=[{**a,'id':a.get('id','EDIT '+a['owner']),'role':a.get('role','texture')} for a in assets]
    commit(base,output,rows,rows[0]['source_size'])
    return output
