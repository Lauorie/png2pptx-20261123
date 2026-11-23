import numpy as np
from PIL import Image,ImageDraw
from full_recovery.source_enclosures import ellipse
from full_recovery.handoff import enclosed_glyph_shapes


def test_texture_cannot_nominate_an_inscribed_ellipse():
    rng=np.random.default_rng(42)
    source=Image.fromarray(rng.integers(0,256,(80,80,3),dtype=np.uint8))
    ImageDraw.Draw(source).rectangle((2,2,77,77),outline='black',width=2)
    assert ellipse(source,[0,0,80,80]) is None
    source=Image.new('RGB',(80,80),'white')
    ImageDraw.Draw(source).ellipse((3,3,76,76),outline='black',width=2)
    assert ellipse(source,[0,0,80,80]) is not None


def test_circled_glyph_owns_only_its_matching_hollow_ring():
    part={'kind':'ellipse','status':'ok','bbox_2d':[20,20,40,40]}
    scene={'parts':{'ring':part}};styles={'ring':{'fill':None,'stroke':'#000000'}}
    row={'id':'label','text':'③','bbox_2d':[20,20,40,40]}
    assert enclosed_glyph_shapes(scene,styles,[row])[0]==['ring']
    assert enclosed_glyph_shapes(scene,styles,[{**row,'text':'3'}])[0]==[]
    assert enclosed_glyph_shapes(scene,styles,[{**row,'bbox_2d':[60,20,80,40]}])[0]==[]
