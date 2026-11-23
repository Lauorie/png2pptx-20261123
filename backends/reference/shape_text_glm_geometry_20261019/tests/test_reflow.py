from copy import deepcopy

from PIL import Image,ImageDraw,ImageFont

from shape_integration.texts.layout import resolve_font
from shape_integration.texts.reflow import propose,refine
from shape_integration.texts.colors import run_consensus


def paragraph():
    lines=['Alpha beta','gamma delta','epsilon zeta']
    image=Image.new('RGB',(220,150),'white');draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((4,2,210,145),radius=30,fill='#D5EAF5')
    font=ImageFont.truetype(resolve_font('Arial',True,False)['path'],24)
    for i,line in enumerate(lines):draw.text((20,12+38*i),line,font=font,fill='#202020')
    text=' '.join(lines)
    row={'id':'P1','text':text,'runs':[{'text':text,'script':'normal','color':'#FF0000'}],
         'font':'Arial','bold':True,'italic':False,'color':'#FF0000','rotation_deg':0,
         'kind':'text','bbox_2d':[3,12,195,123]}
    return row,image,lines


def test_reflow_uses_glyphs_not_predicted_color_and_preserves_characters():
    row,image,lines=paragraph();before=deepcopy(row)
    children,audit=propose(row,image)
    assert audit['status']=='accepted'
    assert [c['text'] for c in children]==lines
    assert all(c['runs'][0]['color']=='#FF0000' for c in children)
    assert len({c['id'] for c in children})==len(lines)
    assert row==before


def test_reflow_cannot_invent_correct_text_for_a_bad_transcription():
    row,image,_=paragraph()
    row['text']='Unrelated content without matching any letters'
    row['runs']=[{'text':row['text'],'script':'normal'}]
    children,audit=propose(row,image)
    assert children is None and audit['status']!='accepted'


def test_formula_stacking_is_not_plain_text_reflow():
    row,image,_=paragraph();row['kind']='formula'
    children,audit=propose(row,image)
    assert children is None and audit['status']=='ineligible_math'


def test_parent_anchor_can_correct_an_incomplete_model_split(tmp_path):
    original,image,lines=paragraph();source=tmp_path/'source.png';image.save(source)
    rows=[]
    for index,text in enumerate((lines[0],' '.join(lines[1:]))):
        row=deepcopy(original)
        row.update(id=f'P1.L{index+1:02d}',parent_id='P1',text=text,
                   parent_geometry_prior_bbox=original['bbox_2d'],
                   bbox_2d=[20,15+index*38,180,40+index*38],
                   runs=[{'text':text,'script':'normal','color':'#FF0000'}])
        rows.append(row)
    result=refine(rows,source,tmp_path/'audit')
    assert [r['text'] for r in result]==lines
    assert all(r['parent_id']=='P1' for r in result)


def test_same_run_consensus_does_not_override_other_color_or_missing_glyphs():
    row={'runs':[{'text':'ABCDE','script':'normal','color':'#CC4400'}],
         'glyph_registration':{'status':'accepted','score':.95}}
    strong={'peak_fraction':.8,'foreground_fraction':.3}
    weak={'peak_fraction':.3,'foreground_fraction':.3}
    samples={0:('#000000',strong),1:('#000000',strong),2:('#000000',weak),
             3:('#000000',weak),4:('#0000FF',strong)}
    result=run_consensus(row,samples)
    assert set(result)=={0,1,2,3}
    row['glyph_registration']['score']=.7
    assert not run_consensus(row,samples)
