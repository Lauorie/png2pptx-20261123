from PIL import Image, ImageDraw
from full_recovery.source_punctuation import candidates, observe
from shape_integration.fitting.paths import ellipse_commands


def test_repeated_symbols_require_complete_named_source_native_components():
    source=Image.new('RGB',(110,70),'white');draw=ImageDraw.Draw(source)
    rows=[{'id':'marks','text':'○ ○','color':'#555555','bbox_2d':[28,28,82,42]}]
    scene={'parts':{}};styles={'parts':{}}
    for i,x in enumerate((35,75)):
        draw.ellipse((x-5,30,x+5,40),outline='#555555',width=1)
        key=f'ring{i}';scene['parts'][key]={'id':key,'object_id':'rings','status':'ok',
            'commands':ellipse_commands([x,35],[5,0],[0,5])}
        styles['parts'][key]={'style':{'stroke':'#555555','width_px':1,'fill':None}}
    assert candidates(rows)==rows
    kept,receipts=observe(source,source,rows,scene,styles,{'arrows':[]},{})
    assert kept==[] and receipts[0]['action']=='retire_duplicate_text'
    del scene['parts']['ring1']
    kept,receipts=observe(source,source,rows,scene,styles,{'arrows':[]},{})
    assert kept==rows
    assert candidates([{**rows[0],'text':'A ○ B'}])==[]
