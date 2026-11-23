from shape_integration.styling.occlusion import discover


def rect(owner,x,y):
    return {'object_id':owner,'kind':'rectangle','status':'ok',
        'commands':[['M',x,y],['L',x+10,y],['L',x+10,y+10],['L',x,y+10],['Z']]}


def line(owner,start,end):
    return {'object_id':owner,'kind':'path','status':'ok',
        'commands':[['M',*start],['L',*end]]}


def test_wire_entering_another_mixed_components_face_is_reviewed():
    table={'left_face':rect('left',10,10),'left_wire':line('left',[20,15],[30,15]),
           'right_face':rect('right',40,10),'right_wire':line('right',[15,15],[40,15])}
    _,relations=discover(table,{'geometry_tolerance_px':.2,'minimum_overlap_area_px':1.})
    assert relations['pending']==[{'a':'left','b':'right','kind':'partial_overlap'}]
    assert not relations['edges']  # nomination never assigns paint order


def test_shared_face_boundary_does_not_create_a_wire_occlusion_question():
    table={'left_face':rect('left',10,10),'left_wire':line('left',[10,15],[5,15]),
           'right_face':rect('right',20,10),'right_wire':line('right',[30,15],[35,15])}
    _,relations=discover(table,{'geometry_tolerance_px':.2,'minimum_overlap_area_px':1.})
    assert not relations['pending'] and not relations['edges']
