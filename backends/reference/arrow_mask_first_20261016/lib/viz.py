"""White-background geometry is separate from CLEAN; no source ink in CURRENT."""
import colorsys
from html import escape
from pathlib import Path
import numpy as np
import cairosvg
from PIL import Image,ImageDraw,ImageFont
from fit import points
from preview_path import path_d,shaft_d,head_trim
from style import pending

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def svg(size,rows,colored=False):
    w,h=size;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    palette={}
    for i,r in enumerate(rows):
        ps=points(r)
        group=r.get('parent_id',r['id'])
        color_index=palette.setdefault(group,len(palette))
        from style_contract import exportable
        s=exportable(r['style']);color=s['color'] if not colored else '#'+''.join(f'{round(v*255):02x}' for v in colorsys.hsv_to_rgb(color_index*.61803398875%1,.85,.8))
        if r.get('kind')=='block_arrow':
            from shape_render import svg_parts
            parts.append(svg_parts({**r,'style':s},color if colored else None,f'block{i}_'))
            continue
        width=s.get('width_px') or r.get('width_estimate') or 2
        defs=[];markers={}
        for end in ['start','end']:
            head=s.get(end)
            if not head:continue
            name=f'h{i}{end}';kind=head['type'];fill=color;outline='none'
            shapes={'triangle':'M0,-1.5 L3,0 L0,1.5 Z','stealth':'M0,-1.5 L3,0 L0,1.5 L.7,0 Z',
                    'arrow':'M0,-1.5 L3,0 L0,1.5','diamond':'M-1,0 L1,-1.2 L3,0 L1,1.2 Z'}
            if kind=='oval':shape=f'<ellipse cx="1.5" cy="0" rx="1.5" ry="1.5" fill="{color}"/>'
            else:
                if kind=='arrow':fill='none';outline=color
                shape=f'<path d="{shapes.get(kind,shapes["triangle"])}" fill="{fill}" stroke="{outline}" stroke-width=".7"/>'
            if r.get('beziers'):
                c=r['beziers'][0 if end=='start' else -1]
                direction=np.subtract(c[0],c[1]) if end=='start' else np.subtract(c[3],c[2])
            else:direction=ps[0]-ps[1] if end=='start' else ps[-1]-ps[-2]
            angle=float(np.degrees(np.arctan2(direction[1],direction[0])))
            # CairoSVG does not implement auto-start-reverse consistently.
            factors={'sm':2/3,'med':1.,'lg':5/3}
            sx=factors[head.get('length','med')];sy=factors[head.get('width','med')]
            shape=f'<g transform="translate(3,0) scale({sx},{sy}) translate(-3,0)">{shape}</g>'
            defs.append(f'<marker id="{name}" markerWidth="12" markerHeight="12" refX="3" refY="0" viewBox="-5 -6 12 12" orient="{angle}" markerUnits="strokeWidth">{shape}</marker>')
            markers[end]=f' marker-{end}="url(#{name})"'
        parts.append('<defs>'+''.join(defs)+'</defs>')
        d=path_d(ps,r.get('beziers'))
        shaft=shaft_d(ps,r.get('beziers'),width,[s.get('start'),s.get('end')])
        dash=s['dash'].get('preset','solid');pattern=f' stroke-dasharray="{4*width} {3*width}"' if dash=='dash' else (f' stroke-dasharray="{width} {2*width}"' if dash=='sysDot' else '')
        if s['dash'].get('mark_px') and s['dash'].get('gap_px'):
            pattern=f' stroke-dasharray="{s["dash"]["mark_px"]} {s["dash"]["gap_px"]}"'
            if 'phase_px' in s['dash']:
                trim=head_trim(float(np.linalg.norm(np.diff(ps,axis=0),axis=1).sum()),width,s.get('start'))
                pattern+=f' stroke-dashoffset="{s["dash"]["phase_px"]+trim}"'
        if s['dash'].get('pattern_px'):
            pattern=' stroke-dasharray="'+' '.join(map(str,s['dash']['pattern_px']))+'"'
        cap='butt'
        parts.append(f'<path id="{escape(r["id"])}" d="{shaft}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="{cap}" stroke-linejoin="miter" stroke-miterlimit="4"{pattern}/>')
        # Invisible original path carries heads at unchanged true endpoints.
        if markers:parts.append(f'<path d="{d}" fill="none" stroke="none" stroke-width="{width}"{markers.get("start","")}{markers.get("end","")}/>')
    return ''.join(parts)+'</svg>'


def render(clean,rows,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);original=Image.open(clean).convert('RGB')
    for name,colored in [('styled',False),('instances',True)]:
        text=svg(original.size,rows,colored);(folder/(name+'.svg')).write_text(text)
        cairosvg.svg2png(bytestring=text.encode(),write_to=str(folder/(name+'_rgba.png')))
        layer=Image.open(folder/(name+'_rgba.png')).convert('RGBA')
        white=Image.new('RGBA',original.size,'white');white.alpha_composite(layer)
        im=white.convert('RGB');draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,14)
        if colored:
            shown=set()
            for r in rows:
                label=r.get('label',r['id'])
                if pending(r):label+='?'
                if label in shown:continue
                shown.add(label)
                ps=points(r);x,y=ps[len(ps)//2]
                draw.text((max(1,min(im.width-38,x+5)),max(1,min(im.height-18,y+5))),label,fill='#333333',font=font,stroke_width=1,stroke_fill='white')
        im.save(folder/(name+'.png'))
        if colored:
            layer.putalpha(layer.getchannel('A').point(lambda a:round(a*.75)))
            overlay=original.convert('RGBA');overlay.alpha_composite(layer);overlay.convert('RGB').save(folder/'overlap.png')


def brief(rows,limit=8):
    output=[]
    for r in rows:
        ps=points(r);inds=np.arange(len(ps)) if r.get('kind')=='block_arrow' else np.linspace(0,len(ps)-1,min(limit,len(ps))).round().astype(int)
        output.append(dict(id=r['id'],kind=r['kind'],points=np.round(ps[inds],1).tolist(),heads=r['heads']))
    return output
