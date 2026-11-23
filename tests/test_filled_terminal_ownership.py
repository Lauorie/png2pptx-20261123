import numpy as np
from full_recovery.quality_choice import filled_routes
from full_recovery.handoff import resample


def fixture(terminal):
    shape={'id':'stem','object_id':'owner','status':'ok','kind':'path','bbox_2d':[48,30,52,44],
           'commands':[['M',48,30],['L',52,30],['L',52,44],['L',48,44],['Z']]}
    parts={'stem':shape};styles={'stem':{'fill':'#AA3366','stroke':None}}
    if terminal:
        parts['dot']={'id':'dot','object_id':'owner','status':'ok','kind':'ellipse','bbox_2d':[46,24,54,32]}
        styles['dot']={'fill':'#AA3366','stroke':None}
    arrow={'id':'arrow:route','points':resample([np.array([[50,28],[50,44]],float)]),
           'heads':['oval','none'],'style':{'color':'#AA3366','width_px':2}}
    return {'parts':parts},styles,[arrow]


def test_short_solid_shaft_with_attached_terminal_is_one_native_comparison():
    scene,styles,arrows=fixture(True);rows=filled_routes(scene,styles,arrows,set())
    assert len(rows)==1 and rows[0]['keys']==['stem','dot']
    assert rows[0]['heads']==['oval','none'] and rows[0]['attachment_refs']==['owner/dot']


def test_short_node_without_terminal_does_not_become_an_arrow():
    scene,styles,arrows=fixture(False)
    assert filled_routes(scene,styles,arrows,set())==[]
