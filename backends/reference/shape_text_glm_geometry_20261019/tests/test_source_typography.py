from PIL import Image, ImageDraw, ImageFont
from shape_integration.texts.line_evidence import line_bands
from shape_integration.texts import font_selection, layout, registration


def sample():
    return dict(id='T1',text='Alpha Beta',font='Times New Roman',bold=False,italic=False,
                color='#000000',kind='text',rotation_deg=0,runs=[{'text':'Alpha Beta','script':'normal'}],
                bbox_2d=[5,5,130,85])


def test_two_baselines_detected_but_underlines_not_counted():
    row=sample();font=ImageFont.truetype(layout.resolve_font('Arial')['path'],24)
    im=Image.new('RGB',(200,100),'white');draw=ImageDraw.Draw(im)
    draw.text((10,5),'Alpha',font=font,fill='black');draw.text((10,45),'Beta',font=font,fill='black')
    assert len(line_bands(row,im))==2
    im=Image.new('RGB',(200,100),'white');draw=ImageDraw.Draw(im)
    draw.text((10,5),'Alpha Beta',font=font,fill='black');draw.line((8,35,160,35),fill='black',width=2)
    assert line_bands(row,im)==[]
    row['kind']='formula'
    assert line_bands(row,im)==[]


def test_source_font_selection_requires_margin_and_accepted_fit(tmp_path,monkeypatch):
    row=sample();source=tmp_path/'source.png';Image.new('RGB',(200,100),'white').save(source)
    monkeypatch.setattr(layout,'resolve_font',lambda font,*args:{'path':font})
    def measure(rows,folder,*args):
        folder.mkdir(parents=True)
        return {r['id']:{'font':r['font']} for r in rows}
    monkeypatch.setattr(layout,'native_metrics',measure)
    scores={'Times New Roman':.70,'Arial':.85,'Calibri':.71,'Courier New':.69}
    monkeypatch.setattr(registration,'register',lambda r,m,im:(r,m,{'status':'accepted','candidates':[{'score':scores[r['font']]}]}))
    selected,metrics=font_selection.select([row],source,tmp_path/'a',None,{})
    assert selected[0]['font']=='Arial' and metrics['T1']['font']=='Arial'
    assert row['font']=='Times New Roman'
    scores['Arial']=.72
    selected,_=font_selection.select([row],source,tmp_path/'b',None,{})
    assert selected[0]['font']=='Times New Roman'
