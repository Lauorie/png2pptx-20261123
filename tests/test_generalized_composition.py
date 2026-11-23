import math
import numpy as np
import pytest
from PIL import Image, ImageDraw


def test_source_locked_cleanup_preserves_nonwhite_rgba_outside_mask():
    from full_recovery.image_edit import source_locked_icon
    source=Image.new('RGBA',(100,100),'white');d=ImageDraw.Draw(source)
    d.ellipse((2,2,97,97),outline='#888888',width=2)
    d.polygon([(50,20),(20,80),(80,80)],fill='#D6D6D6')
    edited=Image.new('RGBA',source.size,'white')
    request={'removal_paths':[{'points':[[0,3],[60,3]],'width_px':2}]}
    result,receipt=source_locked_icon(source,edited,[0,0,100,100],request)
    assert result.getpixel((50,50))==source.getpixel((50,50))
    assert np.array_equal(np.asarray(result)[10:],np.asarray(source)[10:])
    assert receipt['rgba_outside_cleanup_preserved']


@pytest.mark.parametrize('angle',[0,.7,math.pi/2])
@pytest.mark.parametrize('factor',[.7,1.,3.])
def test_short_dashed_double_arrow_is_one_native_candidate(angle,factor):
    from full_recovery.quality_choice import candidates,apply_choices
    origin=np.array([20.,30.]);axis=np.array([math.cos(angle),math.sin(angle)])
    normal=np.array([-axis[1],axis[0]])
    segs=[((0,0),(2,1)),((0,0),(2,-1)),((3,0),(6,0)),
          ((8,0),(11,0)),((14,0),(12,1)),((14,0),(12,-1))]
    commands=[]
    for a,b in segs:
        for op,p in [('M',a),('L',b)]:commands.append([op,*(origin+factor*(p[0]*axis+p[1]*normal))])
    scene={'parts':{'k':{'id':'route','object_id':'group','kind':'path','status':'ok','commands':commands}}}
    rows=candidates(scene,{'k':{'stroke':'#000000','fill':None,'width_px':factor}},[])
    assert len(rows)==1 and rows[0]['compound_stroked_route']
    assert rows[0]['heads']==['open','open']
    plan=apply_choices(rows,[{'candidate_ids':[rows[0]['id']],'keep_ids':[]}])
    assert plan['drop_shape_keys']==['k'] and not plan['shape_subpath_suppression']


def test_disconnected_network_paths_are_not_a_dashed_arrow():
    from full_recovery.compound_routes import recognize
    lines=[np.array([[0,y],[40,30-y]]) for y in [0,5,15,25]]
    assert recognize(lines) is None


def test_source_visible_face_boundary_prevents_pattern_occlusion():
    from shapely.geometry import box,MultiLineString
    from shape_integration.styling.occlusion import order_internal_by_paint
    src=Image.new('RGB',(110,100),'white');d=ImageDraw.Draw(src)
    d.rectangle((30,10,95,75),fill='#EEEEEE',outline='black',width=2)
    d.rectangle((10,30,75,95),fill='#EEEEEE',outline='black',width=2)
    d.line((20,50,60,50),fill='black',width=2)
    geoms={'front':box(10,30,75,95),'ticks':MultiLineString([[(20,50),(60,50)],[(20,60),(60,60)]]),
           'back':box(30,10,95,75)}
    styles={k:{'fill':None if k=='ticks' else '#EEEEEE','stroke':'#000000','width_px':2} for k in geoms}
    scene={'objects':{'obj':{'part_keys':list(geoms)}}}
    rel={'object_order':['obj'],'internal_model_edges':[]}
    out=order_internal_by_paint(scene,{k:{} for k in geoms},rel,styles,geoms,source=src)
    order=out['flat_order']
    assert order.index('back')<order.index('front')<order.index('ticks')
    assert any(e['origin']=='source_visible_face_boundary' for e in out['internal_style_edges'])


@pytest.mark.parametrize('axis',[0,1])
def test_ellipsis_uses_source_points_instead_of_font_metrics(axis):
    from full_recovery.source_ellipsis import observe
    src=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(src)
    positions=[(25+12*i,40) if axis==0 else (40,25+12*i) for i in range(3)]
    for x,y in positions:d.ellipse((x-2,y-2,x+2,y+2),fill='black')
    row={'id':'punct','text':'…' if axis==0 else '⋮','color':'#000000',
         'bbox_2d':[20,35,55,45] if axis==0 else [35,20,45,55],'font':'Times New Roman'}
    result=observe(src,[row])
    assert len(result)==1
    assert np.allclose([dot['center'] for dot in result[0]['dots']],positions)
    row['text']='A … B'
    assert not observe(src,[row])


def test_ellipsis_diameter_accounts_for_antialiased_edge_ink():
    from full_recovery.source_ellipsis import observe
    src=Image.new('RGB',(800,400),'white');d=ImageDraw.Draw(src)
    for x in (240,400,560):d.ellipse((x-27,173,x+27,227),fill='black')
    src=src.resize((100,50),Image.Resampling.LANCZOS)
    marks=observe(src,[{'id':'p','text':'…','color':'#000000','bbox_2d':[25,20,75,30]}])
    assert len(marks)==1
    assert all(6.5<dot['diameter']<7.5 and dot['diameter_method']=='source_ink_area' for dot in marks[0]['dots'])


def test_ellipsis_neighborhood_can_include_a_separate_compact_graphic():
    from full_recovery.source_ellipsis import observe
    source=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(source)
    for y in (30,40,50):draw.ellipse((48,y-2,52,y+2),fill='black')
    draw.polygon([(47,68),(53,68),(50,73)],fill='black')
    marks=observe(source,[{'id':'p','text':'⋮','color':'#000000','bbox_2d':[47,31,53,66]}])
    assert len(marks)==1
    assert np.allclose([dot['center'] for dot in marks[0]['dots']],[(50,30),(50,40),(50,50)])


def test_four_source_dots_cannot_be_reinterpreted_as_one_three_dot_group():
    from full_recovery.source_ellipsis import observe
    source=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(source)
    for y in (30,40,50,60):draw.ellipse((48,y-2,52,y+2),fill='black')
    assert not observe(source,[{'id':'p','text':'⋮','color':'#000000','bbox_2d':[47,27,53,64]}])


def test_native_dot_deletion_uses_the_same_source_observation_as_text_export(tmp_path):
    from full_recovery.handoff import ellipsis_shape_ownership
    source=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(source)
    scene={'parts':{}};styles={}
    for i,x in enumerate((25,37,49)):
        draw.ellipse((x-2,38,x+2,42),fill='black')
        scene['parts'][str(i)]={'id':str(i),'object_id':'owner','status':'ok','bbox_2d':[x-2,38,x+2,42],
            'commands':[['M',x-2,38],['L',x+2,38],['L',x+2,42],['L',x-2,42],['Z']]}
        styles[str(i)]={'fill':'#000000'}
    path=tmp_path/'source.png';source.save(path)
    rows=[{'id':'p','text':'…','color':'#000000','bbox_2d':[20,35,55,45]}]
    arrows=[{'id':'route','start':[20,40],'end':[60,40],'heads':['none','triangle']}]
    dropped,_=ellipsis_shape_ownership(scene,styles,rows,path,arrows)
    assert not dropped


def test_unique_source_triplet_replaces_stretched_hollow_native_copy(tmp_path):
    from full_recovery.handoff import ellipsis_shape_ownership
    source=Image.new('RGB',(120,140),'white');draw=ImageDraw.Draw(source)
    for y in (40,60,80):draw.ellipse((47,y-3,53,y+3),fill='black')
    commands=[]
    for y in (41,69,97):
        commands += [['M',47,y-3],['L',53,y-3],['L',53,y+3],['L',47,y+3],['Z']]
    scene={'parts':{'triplet':{'id':'triplet','object_id':'owner','status':'ok',
        'bbox_2d':[47,38,53,100],'commands':commands}}}
    styles={'triplet':{'fill':'#FFFFFF','stroke':'#000000'}}
    rows=[{'id':'p','text':'⋮','color':'#000000','bbox_2d':[45,35,55,85]}]
    path=tmp_path/'source.png';source.save(path)
    dropped,_=ellipsis_shape_ownership(scene,styles,rows,path)
    assert dropped==['triplet']


@pytest.mark.parametrize('separate',[False,True])
def test_source_unique_triplet_owns_a_complete_copy_shifted_normal_to_its_axis(tmp_path,separate):
    from full_recovery.handoff import ellipsis_shape_ownership
    source=Image.new('RGB',(120,140),'white');draw=ImageDraw.Draw(source)
    for y in (40,60,80):draw.ellipse((47,y-3,53,y+3),fill='black')
    parts={};styles={};commands=[]
    for i,y in enumerate((42,64,86)):
        path=[['M',29,y-3],['L',35,y-3],['L',35,y+3],['L',29,y+3],['Z']]
        commands+=path
        if separate:
            parts[str(i)]={'id':str(i),'object_id':'owner','status':'ok','commands':path}
            styles[str(i)]={'fill':'#FFFFFF','stroke':'#000000'}
    if not separate:
        parts['triplet']={'id':'triplet','object_id':'owner','status':'ok','commands':commands}
        styles['triplet']={'fill':'#FFFFFF','stroke':'#000000'}
    rows=[{'id':'p','text':'⋮','color':'#000000','bbox_2d':[45,35,55,85]}]
    path=tmp_path/'source.png';source.save(path)
    dropped,_=ellipsis_shape_ownership({'parts':parts},styles,rows,path)
    assert dropped==sorted(parts)
    # An original external shaft inside the broader search prevents this
    # displaced-pattern proof; its ink must remain independently owned.
    draw.line((31,45,31,90),fill='black');source.save(path)
    dropped,_=ellipsis_shape_ownership({'parts':parts},styles,rows,path)
    assert not dropped


def line_row(ident,layer,y,start=5,end=95,heads=None):
    return {'id':ident,'layer':layer,'points':np.column_stack([np.linspace(start,end,80),np.full(80,y)]),
            'length':end-start,'style':{'stroke':'#000000','width_px':1},'heads':heads or ['none','none']}


@pytest.mark.parametrize('angle',[0.,.55,math.pi/2])
def test_straight_source_identity_selects_existing_fit_without_changing_geometry(angle):
    from full_recovery.source_straight_routes import resolve
    axis=np.array([math.cos(angle),math.sin(angle)]);normal=np.array([-axis[1],axis[0]])
    ends=np.array([40.,40.])+np.array([0.,90.])[:,None]*axis
    src=Image.new('RGB',(160,160),'white');ImageDraw.Draw(src).line(tuple(ends.reshape(-1)),fill='black')
    rows=[]
    for ident,layer,shift in [('precise','shape',0),('offset','arrows',3)]:
        row=line_row(ident,layer,0);row['points']=ends[0]+np.linspace(0,1,80)[:,None]*(ends[1]-ends[0])+shift*normal
        rows.append(row)
    result=resolve(rows,src)
    assert result['choices'] and result['choices'][0]['keep_ids']==['precise']
    assert np.allclose(rows[1]['points'][0],ends[0]+3*normal)


@pytest.mark.parametrize('mode',['parallel','head','partial'])
def test_straight_source_certificate_preserves_other_information(mode):
    from full_recovery.source_straight_routes import resolve
    src=Image.new('RGB',(110,100),'white');d=ImageDraw.Draw(src);d.line((5,40,95,40),fill='black')
    a=line_row('a','shape',40);b=line_row('b','arrows',43)
    if mode=='parallel':d.line((5,43,95,43),fill='black')
    if mode=='head':b['heads']=['none','triangle']
    if mode=='partial':b=line_row('b','arrows',40,40,95)
    result=resolve([a,b],src)
    assert 'b' not in result['claimed_ids']


def test_punctuation_context_and_low_profile_support_cannot_authorize_branch_specific_deletion():
    from full_recovery.source_straight_routes import resolve
    src=Image.new('RGB',(110,100),'white');ImageDraw.Draw(src).line((5,40,95,40),fill='black')
    rows=[line_row('visible','shape',40),line_row('absent','shape',70),line_row('arrow','arrows',70)]
    for row in rows:row['source_endpoint_context']=[{'role':'source_punctuation','endpoint':1,'text_id':'p'}]
    result=resolve(rows,src)
    assert not result['claimed_ids'] and not result['choices']


def test_source_port_fit_can_replace_unique_wrong_node_centre_connection():
    from full_recovery.source_straight_routes import resolve
    src=Image.new('RGB',(110,100),'white');ImageDraw.Draw(src).line((5,40,95,40),fill='black')
    precise=line_row('precise','arrows',40);wrong=line_row('wrong','shape',40)
    wrong['points'][:,1]=np.linspace(60,20,80)
    for row in [precise,wrong]:row['endpoint_nodes']=['left','right']
    result=resolve([precise,wrong],src)
    assert result['choices'][0]['keep_ids']==['precise']


def source_icon_asset():
    return {'id':'icon','role':'icon','source_box_half_open':[20,20,160,160],
        'edit_context_half_open':[20,20,160,160],
        'native_enclosure':{'center':[90,90],'size':[140,140],'rotation_deg':25,'width_px':2},
        'edit':{'owned_box':[20,20,160,160]},
        'edit_request':{'source_pixels_policy':'explicit_cleanup_masks.1','native_labels':[],'removal_paths':[]}}


@pytest.mark.parametrize('control',['external','corner','cleanup','headed','generated','foreground'])
def test_source_image_coverage_preserves_external_and_uncovered_native_information(control):
    from full_recovery.image_line_scope import resolve
    a=source_icon_asset();row=line_row('route','arrows',85,45,130)
    if control=='external':row=line_row('route','arrows',85,5,130)
    if control=='corner':row=line_row('route','arrows',25,25,40)
    if control=='cleanup':a['edit_request']['native_labels']=[{'box':[50,60,75,70]}]
    if control=='headed':row['heads']=['none','triangle']
    if control=='generated':a['edit_request'].pop('source_pixels_policy')
    if control=='foreground':a['edit']['separate_foreground_parts']=['separate']
    result=resolve([row],[a])
    assert not result['claimed_ids']


def test_source_locked_internal_lines_are_owned_even_without_any_shape_competitor(tmp_path):
    from full_recovery.source_ownership import review
    from full_recovery.source_line_coverage import guard
    src=Image.new('RGB',(180,180),'white');ImageDraw.Draw(src).line((45,85,130,85),fill='black')
    source=tmp_path/'source.png';src.save(source)
    row={**line_row('route','arrows',85,45,130),'key':'route','refs':['route'],'keys':[]}
    class NoModel:
        def call(self,*args,**kwargs):raise AssertionError('Original image already covers this exact protected interior')
    out=review({'source_size':[180,180],'parts':{}},{},[row],source,None,None,tmp_path/'review',NoModel(),assets=[source_icon_asset()])
    assert out['drop_arrow_names']==['route'] and not out['source_coverage_guards']
    assert out['source_receipts'][0]['classification']=='source_covered_image_interior'
    assert out['model_calls']==0


def test_background_image_cannot_claim_a_unique_external_arrow(tmp_path):
    from full_recovery.source_ownership import review
    src=Image.new('RGB',(180,180),'white');source=tmp_path/'source.png';src.save(source)
    row={**line_row('route','arrows',85,45,130),'key':'route','refs':['route'],'keys':[]}
    class NoModel:
        def call(self,*args,**kwargs):raise AssertionError('A background cannot expand line ownership')
    asset={**source_icon_asset(),'role':'background'}
    result=review({'source_size':[180,180],'parts':{}},{},[row],source,None,None,tmp_path/'review',NoModel(),assets=[asset])
    assert not result['drop_arrow_names'] and result['source_receipts'][0]['classification']=='outside_shape_scope'


@pytest.mark.parametrize('mode',['partial','parallel','missing_head','source_visible'])
def test_arrow_deletion_requires_its_visible_content_and_both_heads(mode):
    from full_recovery.source_line_coverage import guard
    src=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(src);d.line((5,40,95,40),fill='black')
    a=line_row('a','arrows',40);s=line_row('s','shape',40)
    kept=['s']
    if mode=='partial':s=line_row('s','shape',40,40,80)
    if mode=='parallel':s=line_row('s','shape',45);d.line((5,45,95,45),fill='black')
    if mode=='missing_head':a['heads']=['triangle','triangle'];s['heads']=['none','triangle']
    if mode=='source_visible':kept=[]
    choices,audit=guard([a,s],[{'candidate_ids':['a','s'],'keep_ids':kept}],src)
    assert 'a' in choices[0]['keep_ids'] and audit


def test_complete_source_supported_replacement_can_remove_duplicate_arrow():
    from full_recovery.source_line_coverage import guard
    src=Image.new('RGB',(100,100),'white');ImageDraw.Draw(src).line((5,40,95,40),fill='black')
    choices,audit=guard([line_row('a','arrows',40),line_row('s','shape',40)],
        [{'candidate_ids':['a','s'],'keep_ids':['s']}],src)
    assert choices[0]['keep_ids']==['s'] and not audit


@pytest.mark.parametrize('radius',[20,45,80])
def test_simple_source_enclosure_is_independent_of_complex_interior(radius):
    from full_recovery.source_enclosures import ellipse,clip_interior
    size=2*radius+12;src=Image.new('RGB',(size,size),'white');d=ImageDraw.Draw(src)
    d.ellipse((5,5,size-6,size-6),outline='#777777',width=2)
    d.polygon([(radius,15),(15,radius+10),(radius+12,radius+10)],fill='#BBBBBB')
    enclosure=ellipse(src,[3,3,size-3,size-3])
    assert enclosure and enclosure['source_rim_coverage']>=.88
    clipped=clip_interior(src,[0,0,size,size],enclosure)
    assert clipped.getpixel((0,0))[3]==0
    assert clipped.getpixel((radius,radius))[3]==255
    other=Image.new('RGB',(size,size),'white');ImageDraw.Draw(other).rectangle((5,5,size-6,size-6),outline='black',width=2)
    assert ellipse(other,[3,3,size-3,size-3]) is None


def test_image_edit_is_deferred_until_source_ownership(tmp_path):
    from full_recovery.image_assets import review
    from full_recovery.image_realization import realize
    from full_recovery.compose import sha
    src=tmp_path/'source.png';im=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(im)
    d.rectangle((20,20,80,80),fill='blue');d.text((30,30),'x',fill='black');im.save(src)
    scene={'source_size':[100,100],'source_sha256':sha(src),'parts':{},'objects':{}}
    registry=[{'id':'asset','category':'illustration','target':'blue glyph','source_box_half_open':[15,15,85,85],
               'bbox_2d':[15,15,85,85]}]
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'icon','reason':'label cleanup',
            'replace_parts':[],'edit_target':'blue glyph'})
    class Editor:
        calls=0
        def edit(self,crop,prompt,path,*,role):
            self.calls+=1;crop.save(path)
            return {'source_offset_xy':[0,0],'source_extent_wh':list(crop.size)}
    editor=Editor()
    texture=review(scene,{},[],[{'id':'label','text':'x','bbox_2d':[30,30,35,38]}],src,src,tmp_path/'review',Runtime(),images=registry,editor=editor,defer_cleanup=True)
    assert texture['assets'] and editor.calls==0 and texture['assets'][0]['pending_edit']
    realize(texture,scene,{},src,{'quality_choice':{'source_receipts':[]}},editor)
    assert editor.calls==1 and not texture['assets'][0]['pending_edit']
    assert texture['assets'][0]['edit']['source_ownership_finalized_before_edit']


def test_failed_background_edit_restores_native_membership(tmp_path):
    from full_recovery.image_realization import realize
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    asset={'id':'unit','owner':'owner','role':'background','pending_edit':True,'path':str(tmp_path/'image.png'),
        'edit_context_half_open':[0,0,100,100],'edit_request':{'target':'background'},
        'replace_parts':['body'],'replace_names':['owner/body'],'registry_ids':[]}
    texture={'assets':[asset],'drop_shape_keys':['body'],'drop_shape_names':['owner/body']}
    class Editor:
        def edit(self,*args,**kwargs):raise ConnectionError('offline')
    realize(texture,{'source_size':[100,100],'parts':{}},{},source,{},Editor())
    assert texture['assets']==[] and texture['drop_shape_keys']==[] and texture['drop_shape_names']==[]
    assert texture['realization_failures'][0]['fallback']=='retain_native_owner'


def test_wrong_white_paint_cannot_replace_a_visible_black_arrow():
    from full_recovery.source_line_coverage import guard
    source=Image.new('RGB',(100,100),'white');ImageDraw.Draw(source).line((5,40,95,40),fill='black')
    a=line_row('a','arrows',40);s=line_row('s','shape',40);s['style']['stroke']='#FFFFFF'
    choices,audit=guard([a,s],[{'candidate_ids':['a','s'],'keep_ids':['s']}],source)
    assert 'a' in choices[0]['keep_ids'] and audit


def test_native_dash_gaps_cannot_claim_full_source_coverage():
    from full_recovery.source_line_coverage import guard
    source=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(source);d.line((5,40,95,40),fill='black')
    native=Image.new('RGB',(100,100),'white');nd=ImageDraw.Draw(native)
    segments=[]
    for x in range(5,95,4):
        nd.line((x,40,x+2,40),fill='black');segments.append(np.array([[x,40],[x+2,40]]))
    a=line_row('a','arrows',40);s=line_row('s','shape',40);s['paths']=segments
    choices,audit=guard([a,s],[{'candidate_ids':['a','s'],'keep_ids':['s']}],source,
                        native_views={'shape':np.asarray(native)})
    assert 'a' in choices[0]['keep_ids'] and audit[0]['replacement_coverage']<.9


def test_verified_surface_pattern_owns_duplicate_but_not_a_crossing_arrow(tmp_path):
    from full_recovery.surface_patterns import resolve
    from full_recovery.handoff import strokes
    source=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(source)
    d.rectangle((10,10,90,90),fill='#EEEEEE')
    commands=[]
    for y in [30,45,60,75]:
        d.line((20,y,80,y),fill='black',width=1);commands.extend([['M',20,y],['L',80,y]])
    source.save(tmp_path/'native.png')
    scene={'parts':{
        'face':{'id':'face','object_id':'unit','kind':'rectangle','status':'ok','bbox_2d':[10,10,90,90],
                'commands':[['M',10,10],['L',90,10],['L',90,90],['L',10,90],['Z']]},
        'pattern':{'id':'pattern','object_id':'unit','kind':'path','status':'ok','bbox_2d':[20,30,80,75],'commands':commands}}}
    styles={'face':{'fill':'#EEEEEE','stroke':None},'pattern':{'fill':None,'stroke':'#000000','width_px':1}}
    a=line_row('arrow:a','arrows',45,20,80);a['key']='a'
    b=line_row('arrow:b','arrows',45,0,99);b['key']='b'
    result=resolve(scene,styles,[a,b],source,tmp_path/'shape.pptx')
    assert result['pattern_keys']==['pattern'] and result['drop_arrow_names']==['a']
    Image.new('RGB',(100,100),'white').save(tmp_path/'native.png')
    # Retained-but-hidden geometry cannot authorize deletion.
    assert resolve(scene,styles,[a],source,tmp_path/'shape.pptx')['pattern_keys']==[]


def test_verified_image_interior_hides_only_source_occluded_route(tmp_path):
    from full_recovery.asset_occlusion import reconcile
    source=tmp_path/'source.png';im=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(im)
    d.line((1,50,20,50),fill='black',width=1);d.ellipse((20,20,80,80),outline='#777777',width=1)
    im.save(source)
    asset={'id':'icon','role':'icon','native_enclosure':{'center':[50,50],'size':[60,60],
            'rotation_deg':0,'width_px':1}}
    arrow={'id':'external','start':[1,50],'end':[70,50],'heads':['none','none'],
           'style':{'color':'#000000','width_px':1}}
    placement,_=reconcile([asset],[arrow],source,[])
    assert placement['external']['before']==['IMG icon']
    d.line((1,50,80,50),fill='black',width=1);im.save(source)
    assert reconcile([asset],[arrow],source,[])[0]=={}
    # A crop rectangle has no authority to hide a source connection.
    assert reconcile([{'id':'box','role':'icon','source_box_half_open':[20,20,80,80]}],[arrow],source,[])[0]=={}


def test_bounded_critic_budget_and_failed_rows_only_correction(tmp_path):
    from shape_integration.styling import options
    from shape_integration.styling import model
    class Runtime:
        def __init__(self):self.calls=[];self.overrides={}
        def for_stage(self,**kwargs):self.overrides=kwargs;return self
        def call(self,folder,tag,system,prompt,images,validator,budget):
            self.calls.append(([t['id'] for t in prompt['targets']],budget))
            return validator({'checks':[{'id':t['id'],'verdict':('invalid' if len(self.calls)==1 and t['id']=='P1' else 'accept')}
                                        for t in prompt['targets']]})
    rt=Runtime();opt=options({'styling':{'max_tokens':100000}})
    prompt=model.prompt('critic',[{'id':'P0'},{'id':'P1'}],{})
    result=model.ask(rt,tmp_path,'critic',prompt,[],['P0','P1'],{'P0':{},'P1':{}},opt)
    assert set(result['valid'])=={'P0','P1'}
    assert [ids for ids,budget in rt.calls]==[['P0','P1'],['P1']]
    assert all(budget==8192 for ids,budget in rt.calls)
    assert rt.overrides['reasoning_effort']=='low'


def test_registered_source_without_competitor_skips_both_revotes_and_generation(tmp_path):
    from full_recovery.image_assets import review
    from full_recovery.image_realization import realize
    from full_recovery.compose import sha
    source=tmp_path/'source.png';image=Image.new('RGB',(100,100),'white');ImageDraw.Draw(image).rectangle((20,20,80,80),fill='blue');image.save(source)
    scene={'source_size':[100,100],'source_sha256':sha(source),'parts':{},'objects':{}}
    registry=[{'id':'asset','category':'illustration','target':'blue glyph','source_box_half_open':[15,15,85,85],
               'bbox_2d':[15,15,85,85]}]
    class Forbidden:
        def call(self,*args,**kwargs):raise AssertionError('No native representation choice exists')
        def edit(self,*args,**kwargs):raise AssertionError('No source pixels need editing')
    texture=review(scene,{},[],[],source,source,tmp_path/'review',Forbidden(),images=registry,
                   editor=Forbidden(),defer_cleanup=True,preserve_icon_pixels=True)
    realize(texture,scene,{},source,{},Forbidden())
    assert texture['model_calls']==0 and texture['assets'][0]['edit']['requests']==0
    with Image.open(texture['assets'][0]['path']) as actual:
        assert np.array_equal(np.asarray(actual),np.asarray(image.convert('RGBA').crop((15,15,85,85))))
