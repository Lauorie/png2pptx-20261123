import copy
from pathlib import Path
import pytest
from PIL import Image,ImageDraw
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from full_recovery.image_assets import review,commit,owner_matches,clip_source_edge,clip_owned_crop


def test_containing_frame_cannot_consume_registered_interior():
    scene={'source_size':[200,200],
        'objects':{'frame':{'candidate':{'bbox_2d':[20,20,180,180],
            'label':'ordinary circular frame','kind':'ellipse'}}},
        'parts':{'frame':{'object_id':'frame','bbox_2d':[20,20,180,180]}}}
    images=[{'id':'interior','source_box_half_open':[35,35,165,165],
             'target':'complex original interior'}]
    matched,unmatched=owner_matches(scene,images)
    assert not matched and unmatched==images


@pytest.mark.parametrize('incomplete',[False,True])
def test_partial_container_enters_existing_review_without_replacing_correct_parts(tmp_path,incomplete):
    source=tmp_path/'source.png';im=Image.new('RGB',(200,200),'white');draw=ImageDraw.Draw(im)
    draw.polygon([(50,50),(100,60),(60,90)],fill='blue');im.save(source)
    frame={'id':'frame','object_id':'owner','kind':'rectangle','status':'ok','bbox_2d':[20,20,150,150],
        'commands':[['M',20,20],['L',150,20],['L',150,150],['L',20,150],['Z']],
        'native_parameters_original':{'center':[85,85],'width':130,'height':130}}
    dot={'id':'dot','object_id':'owner','kind':'ellipse','status':'ok','bbox_2d':[80,120,90,130],
        'commands':[['M',80,120],['L',90,120],['L',90,130],['L',80,130],['Z']]}
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':{'frame':frame,'dot':dot},
        'objects':{'owner':{'candidate':{'label':'ordinary container','kind':'rectangle','bbox_2d':[20,20,150,150]},
            'part_keys':['frame','dot'],'completeness_status':'incomplete' if incomplete else 'unreviewed',
            'observation_status':'partial' if incomplete else 'ok','selected_revision':'revision_01',
            'history':[{'directory':'revision_00','observation':{'reason':'stale reason'}},
                       {'directory':'revision_01','observation':{'reason':'Interior illustration omitted'}}]}}}
    original=copy.deepcopy(scene)
    class Runtime:
        def call(self,*args):
            evidence=args[3]['incomplete_native_target']
            assert evidence['recorded_reason']=='Interior illustration omitted'
            assert next(p for p in args[3]['parts'] if p['key']=='P0')['protected']
            return args[5]({'action':'image','method':'crop','role':'icon','reason':'Only the missing interior',
                'replace_parts':[],'edit_box':[50,50,101,91]})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime())
    assert result['model_calls']==int(incomplete)
    assert result['drop_shape_keys']==[] and result['drop_arrow_names']==[] and scene==original
    assert result['incomplete_owner_candidates']==(['owner'] if incomplete else [])
    if incomplete:
        assert len(result['assets'])==1
        assert result['assets'][0]['placement_source_half_open']==[50,50,101,91]
        assert result['assets'][0]['replace_parts']==[]
    else:assert result['assets']==[]


@pytest.mark.parametrize('label',['network','diagram','icon','ordinary object'])
def test_image_owner_nomination_uses_native_footprint_instead_of_label(label):
    scene={'source_size':[200,200],
        'objects':{'owner':{'candidate':{'bbox_2d':[30,30,150,150],
            'label':label,'kind':'compound_symbol'}}},
        'parts':{'inside':{'object_id':'owner','bbox_2d':[40,40,140,140]}}}
    images=[{'id':'interior','source_box_half_open':[35,35,145,145]}]
    matched,unmatched=owner_matches(scene,images)
    assert matched=={'owner':images} and not unmatched


def test_editor_padding_cleanup_preserves_content_and_true_clipped_components():
    from full_recovery.image_edit import transparent_exterior,clear_padding_fragments
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((30,30,70,70),fill='black')
    draw.rectangle((90,98,99,99),fill=(150,145,145))
    cleaned,count=clear_padding_fragments(transparent_exterior(image),[20,20,80,80])
    assert count==20 and cleaned.getchannel('A').getbbox()==(30,30,71,71)
    draw.line((50,50,99,50),fill='black',width=2)
    draw.rectangle((0,0,3,3),fill='red')
    cleaned,count=clear_padding_fragments(transparent_exterior(image),[20,20,80,80])
    assert count==20
    assert cleaned.getpixel((99,50))[3]==255  # a real clipped contour is still rejected downstream
    assert cleaned.getpixel((0,0))[3]==255  # colored edge content is never treated as neutral padding


def test_known_editor_padding_seeds_exterior_inside_a_sealed_paper_rim():
    from full_recovery.image_edit import transparent_exterior,clear_padding_fragments
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,99,99),outline=(160,160,160),width=2)
    draw.rectangle((30,30,70,70),outline='black',width=3)
    box=[20,20,80,80]
    result,count=clear_padding_fragments(transparent_exterior(image,content_box=box),box)
    assert count>0 and result.getchannel('A').getbbox()==(30,30,71,71)
    assert result.getpixel((50,50))==(255,255,255,255)
    # A real object connected to the edge must still fail the canvas check.
    draw.line((50,50,99,50),fill='black',width=2)
    result,_=clear_padding_fragments(transparent_exterior(image,content_box=box),box)
    assert result.getpixel((99,50))[3]==255


def test_multi_cell_image_scope_excludes_native_grid_between_cells():
    from full_recovery.image_assets import clip_asset_scope
    def part(b):
        x,y,r,t=b
        return {'bbox_2d':b,'commands':[['M',x,y],['L',r,y],['L',r,t],['L',x,t],['Z']]}
    scene={'parts':{'left':part([5,5,40,40]),'right':part([60,5,95,40]),
                    'a':part([10,10,30,30]),'b':part([70,10,90,30])}}
    result=clip_asset_scope(Image.new('RGBA',(100,50),'red'),[0,0,100,50],[0,0,100,50],scene,{},
        frames=['left','right'],owned_parts=['a','b'])
    assert result.getpixel((20,20))[3]==255 and result.getpixel((80,20))[3]==255
    assert result.getpixel((50,20))[3]==0 and result.getpixel((5,20))[3]==0


def test_edit_scope_keeps_simple_enclosure_and_native_text(tmp_path):
    from full_recovery.image_assets import native_frames
    source=tmp_path/'source.png';Image.new('RGB',(1000,1000),'white').save(source)
    frame={'id':'frame','object_id':'icon','kind':'rectangle','status':'ok','bbox_2d':[10,10,90,90],
        'commands':[['M',10,10],['L',90,10],['L',90,90],['L',10,90],['Z']],
        'native_parameters_original':{'center':[50,50],'width':80,'height':80}}
    glyph={'id':'glyph','object_id':'icon','kind':'path','status':'ok','bbox_2d':[30,40,60,60],
        'commands':[['M',30,40],['L',60,60]]}
    scene={'source_size':[1000,1000],'source_sha256':sha(source),'parts':{'frame':frame,'glyph':glyph},
        'objects':{'icon':{'candidate':{'label':'small icon','kind':'compound_symbol','bbox_2d':[10,10,90,90]}}}}
    class Runtime:
        def call(self,*args):
            assert next(p for p in args[3]['parts'] if p['key']=='P0')['protected']
            return args[5]({'action':'image','method':'edit','role':'icon','reason':'interior only',
                'replace_parts':['P1'],'edit_target':'Recover the interior symbol only.'})
    class Editor:
        def edit(self,crop,prompt,path,*,role):
            assert crop.size==(30,20)
            Image.new('RGBA',(50,40),'red').save(path)
            return {'source_offset_xy':[-10,-10],'source_extent_wh':[50,40]}
    assert native_frames(scene,list(scene['parts']))==['frame']
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),selected=['icon'],editor=Editor())
    assert result['drop_shape_keys']==['glyph']
    asset=result['assets'][0];im=Image.open(asset['path'])
    assert im.getchannel('A').getbbox()==(10,10,40,30)
    p=Presentation();p.slide_width=p.slide_height=1000000;s=p.slides.add_slide(p.slide_layouts[6])
    border=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,100000,100000,800000,800000);border.name='icon/frame'
    old=s.shapes.add_connector(1,300000,400000,600000,600000);old.name='icon/glyph'
    text=s.shapes.add_textbox(350000,150000,300000,150000);text.name='TXT title';text.text='unchanged'
    original=[border._element.xml,text._element.xml]
    p.save(tmp_path/'before.pptx');commit(tmp_path/'before.pptx',tmp_path/'after.pptx',[asset],[100,100])
    shapes=Presentation(tmp_path/'after.pptx').slides[0].shapes
    assert [v._element.xml for v in shapes if v.name in ('icon/frame','TXT title')]==original


def test_registered_edit_without_explicit_cleanup_preserves_original_pixels(tmp_path):
    source,scene,registered=fixture(tmp_path);scene['objects']={};scene['parts']={}
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'icon',
            'reason':'only interior needs cleaning','replace_parts':[],
            'edit_box':[30,30,60,60],'edit_target':'Clean only this interior.'})
    class Editor:
        def edit(self,crop,prompt,path,*,role):
            Image.new('RGBA',crop.size,'red').save(path)
            return {'source_offset_xy':[0,0],'source_extent_wh':list(crop.size)}
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered,editor=Editor())
    asset=result['assets'][0];im=Image.open(asset['path'])
    assert asset['placement_source_half_open']==[10,10,105,90]
    assert im.getpixel((2,2))[:3]==(0,0,255)
    assert im.getpixel((30,30))[:3]==(0,0,255)
    assert asset['edit']['source_pixels_preserved']
    assert asset['edit']['edited_fraction']==0
    assert asset['edit']['unedited_registry_pixels_preserved']


@pytest.mark.parametrize('partial',[False,True])
def test_edited_label_background_is_not_punched_out_again(tmp_path,partial):
    source,scene,registered=fixture(tmp_path);scene['objects']={};scene['parts']={}
    requested=[30,30,60,60] if partial else [10,10,105,90]
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'icon',
            'reason':'remove native label, keep coloured body','replace_parts':[],
            'edit_box':requested,'edit_target':'Keep the original coloured body.'})
    class Editor:
        def edit(self,crop,prompt,path,*,role):
            assert prompt['native_labels']
            Image.new('RGBA',(crop.width*2,crop.height*2),'red').save(path)
            return {'source_offset_xy':[0,0],'source_extent_wh':list(crop.size)}
    result=review(scene,{},[],[{'text':'X','bbox_2d':[35,35,45,45]}],source,source,
        tmp_path/'review',Runtime(),images=registered,editor=Editor())
    asset=result['assets'][0]
    with Image.open(asset['path']) as im:
        scale=im.width/(asset['placement_source_half_open'][2]-asset['placement_source_half_open'][0])
        assert im.getpixel((round(30*scale),round(30*scale)))==(255,0,0,255)
        if partial:assert im.getpixel((2,2))[:3]==(0,0,255)
        else:assert im.size==(95,80)  # registered assets keep original source pixels and extent
        assert asset['edit']['source_pixels_preserved']
    assert bool(asset['edit'].get('unedited_registry_pixels_preserved'))==partial


def test_exact_observation_crop_means_whole_target_edit_without_expanding_ownership(tmp_path):
    source,scene,registered=fixture(tmp_path);scene['objects']={};scene['parts']={}
    class Runtime:
        def call(self,*args):
            prompt=args[3]
            return args[5]({'action':'image','method':'edit','role':'icon',
                'reason':'clean the complete registered target','replace_parts':[],
                'edit_box':prompt['source_box'],'edit_target':'Preserve the source body.'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered,
                  editor=object(),defer_cleanup=True)
    assert len(result['assets'])==1
    asset=result['assets'][0]
    assert asset['placement_source_half_open']==[10,10,105,90]
    assert asset['edit_request']['edit_box'] is None


def test_partial_registered_edit_cannot_leave_native_text_in_retained_pixels(tmp_path):
    source,scene,registered=fixture(tmp_path);scene['objects']={};scene['parts']={}
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'icon',
            'reason':'local cleanup','replace_parts':[],'edit_box':[30,30,60,60],
            'edit_target':'Keep the coloured body.'})
    class Editor:
        def edit(self,*args,**kwargs):raise AssertionError('Scope must be resolved before editing')
    result=review(scene,{},[],[{'text':'X','bbox_2d':[80,35,90,45]}],source,source,
        tmp_path/'review',Runtime(),images=registered,editor=Editor())
    assert not result['assets']
    assert 'include native labels' in result['reviews'][0]['reason']


def test_narrow_edit_cannot_delete_a_whole_native_target(tmp_path):
    source,scene,registered=fixture(tmp_path)
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'background',
            'replace_parts':['body'],'edit_box':[10,10,105,20],
            'edit_target':'blue cloud','reason':'clean only the top strip'})
    class Editor:
        def edit(self,*args,**kwargs):raise AssertionError('Reject destructive scope before editing')
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered,editor=Editor())
    assert not result['assets'] and not result['drop_shape_keys']
    assert result['reviews'][0]['action']=='keep_native'
    assert 'complete replace_parts' in result['reviews'][0]['reason']
from shape_integration.common import sha


def test_image_replaces_foreground_symbol_above_interleaved_card_without_reordering_neighbors(tmp_path):
    asset_path=tmp_path/'icon.png';Image.new('RGBA',(10,10),'black').save(asset_path)
    deck=Presentation();deck.slide_width=deck.slide_height=1000000
    slide=deck.slides.add_slide(deck.slide_layouts[6])
    for name in ['icon/mouth','icon/card','icon/eyes','foreground/label']:
        item=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,100000,100000,500000,500000);item.name=name
    protected=[slide.shapes[i]._element.xml for i in (1,3)]
    deck.save(tmp_path/'before.pptx')
    asset={'id':'one','owner':'icon','role':'icon','path':str(asset_path),'sha256':sha(asset_path),
        'placement_source_half_open':[10,10,60,60],'replace_names':['icon/mouth','icon/eyes']}
    commit(tmp_path/'before.pptx',tmp_path/'after.pptx',[asset],[100,100])
    shapes=Presentation(tmp_path/'after.pptx').slides[0].shapes
    assert [s.name for s in shapes]==['icon/card','IMG one','foreground/label']
    assert [shapes[i]._element.xml for i in (0,2)]==protected


def test_crop_frame_excludes_external_connector_padding_but_keeps_internal_marks():
    im=Image.new('RGBA',(100,100),(255,255,255,0));d=ImageDraw.Draw(im)
    d.rectangle((10,10,85,75),outline='black',width=2)
    d.line((20,20,70,60),fill='blue',width=2)
    d.line((50,78,50,99),fill='black',width=3)
    frame={'commands':[['M',10,10],['L',85,10],['L',85,75],['L',10,75],['Z']]}
    inside={'commands':[['M',20,20],['L',70,60]]}
    scene={'parts':{'frame':frame,'inside':inside}}
    result,key=clip_owned_crop(im,[0,0,100,100],scene,{'frame':{'width_px':2}},list(scene['parts']))
    assert key=='frame' and result.getpixel((50,85))[3]==0
    assert result.getpixel((20,20))==im.getpixel((20,20))
    assert result.getpixel((85,75))==im.getpixel((85,75))
    scene['parts']['external']={'commands':[['M',50,78],['L',50,99]]}
    same,key=clip_owned_crop(im,[0,0,100,100],scene,{},list(scene['parts']))
    assert key is None and same.tobytes()==im.tobytes()


def test_white_editor_paper_rim_is_removed_without_erasing_colored_edges_or_white_interiors():
    from full_recovery.image_edit import transparent_exterior
    im=Image.new('RGB',(100,100),(245,245,245));d=ImageDraw.Draw(im)
    d.line((0,99,99,99),fill=(190,190,190),width=1)
    d.rectangle((20,20,80,80),outline='navy',width=3)
    result=transparent_exterior(im)
    assert result.getchannel('A').getbbox()==(20,20,81,81)
    assert result.getpixel((50,50))[3]==255
    d.line((0,40,30,40),fill='navy',width=2)
    result=transparent_exterior(im)
    assert result.getpixel((0,40))[3]==255


def fixture(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(160,120),'white')
    draw=ImageDraw.Draw(im);draw.rectangle((10,10,105,90),fill='blue')
    draw.ellipse((40,30,50,40),fill='green');im.save(source)
    part={'id':'body','object_id':'cloud','kind':'path','status':'ok','bbox_2d':[10,10,105,90],
          'commands':[['M',10,10],['L',105,10],['L',105,90],['L',10,90],['Z']]}
    scene={'source_size':[160,120],'source_sha256':sha(source),'parts':{'body':part},
           'objects':{'cloud':{'candidate':{'label':'cloud backdrop','bbox_2d':[10,10,105,90],'kind':'cloud'}},
                      'network':{'candidate':{'label':'node connection network','bbox_2d':[10,10,105,90]}}}}
    registered=[{'id':'I0001','target':'cloud background','category':'decoration','source_box_half_open':[10,10,105,90],
        'path':str(source),'sha256':sha(source),'source_sha256':sha(source),'remove':'foreground nodes and lines'}]
    return source,scene,registered


def test_registered_background_has_one_decision_and_cannot_return_as_old_opaque_crop(tmp_path):
    source,scene,registered=fixture(tmp_path)
    # A containing network is not a second candidate owner for its backdrop.
    matches,unmatched=owner_matches(scene,registered)
    assert list(matches)==['cloud'] and not unmatched
    class Runtime:
        calls=0
        def call(self,*args):
            self.calls+=1
            return args[5]({'action':'image','method':'edit','role':'background',
                'replace_parts':['body'],'edit_target':'Keep blue cloud; remove foreground nodes and lines.','reason':'wrong contour'})
    class Editor:
        def edit(self,crop,prompt,path,*,role):
            assert role=='background'
            Image.new('RGBA',(80,60),(0,0,255,255)).save(path)
            return {'source_offset_xy':[4,4],'source_extent_wh':[80,60]}
    runtime=Runtime()
    result=review(scene,{},[],[],source,source,tmp_path/'review',runtime,images=registered,editor=Editor())
    assert runtime.calls==1 and len(result['assets'])==1
    assert result['assets'][0]['registry_ids']==['I0001']
    assert result['drop_shape_keys']==['body']
    assert result['registry_ids_consumed']==['I0001']
    p=Presentation();p.slide_width=1600000;p.slide_height=1200000;s=p.slides.add_slide(p.slide_layouts[6])
    node=s.shapes.add_shape(MSO_SHAPE.OVAL,400000,300000,100000,100000);node.name='network/node'
    bg=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,100000,100000,950000,800000);bg.name='cloud/body'
    text=s.shapes.add_textbox(50000,10000,300000,80000);text.name='TXT t';text.text='label'
    line=s.shapes.add_connector(1,400000,300000,800000,300000);line.name='network/line'
    before=[x._element.xml for x in (node,text,line)]
    p.save(tmp_path/'native.pptx')
    commit(tmp_path/'native.pptx',tmp_path/'result.pptx',result['assets'],scene['source_size'])
    shapes=Presentation(tmp_path/'result.pptx').slides[0].shapes
    assert shapes[0].name=='IMG U_cloud'
    assert [s._element.xml for s in list(shapes)[1:]]==before
    assert len(shapes)==4


def test_failed_edit_keeps_native_without_reinserting_registered_opaque_crop(tmp_path):
    source,scene,registered=fixture(tmp_path)
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'background',
            'replace_parts':['body'],'edit_target':'isolate background','reason':'bad silhouette'})
    class Editor:
        def edit(self,*args,**kwargs):raise ValueError('invalid edited alpha')
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered,editor=Editor())
    assert not result['assets'] and not result['drop_shape_keys']
    assert result['registry_ids_consumed']==['I0001']
    assert 'asset_error' in result['reviews'][0]


def test_image_without_explicit_replacement_scope_cannot_delete_an_owner(tmp_path):
    source,scene,registered=fixture(tmp_path)
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'crop','role':'texture',
            'reason':'interior needs repair'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered)
    assert not result['assets'] and not result['drop_shape_keys']
    assert result['reviews'][0]['action']=='keep_native'
    assert 'replace_parts' in result['reviews'][0]['reason']


def test_partial_registered_edit_does_not_restore_separately_owned_foreground(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(140,100),'white')
    draw=ImageDraw.Draw(im);draw.rectangle((20,20,100,80),fill='blue')
    draw.rectangle((40,10,60,45),fill='#00FF00');im.save(source)
    part={'id':'node','object_id':'node','kind':'rectangle','status':'ok','bbox_2d':[40,10,60,45],
        'commands':[['M',40,10],['L',60,10],['L',60,45],['L',40,45],['Z']]}
    scene={'source_size':[140,100],'source_sha256':sha(source),'parts':{'node':part},
        'objects':{'node':{'candidate':{'label':'node','kind':'rectangle','bbox_2d':part['bbox_2d']}}}}
    registry=[{'id':'image','target':'surface','category':'decoration','source_box_half_open':[20,20,100,80]}]
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'texture',
            'replace_parts':[],'edit_box':[20,40,100,80],'edit_target':'Restore only the texture.','foreground_parts':['O0'],
            'reason':'clean the interior, preserve the exterior'})
    class Editor:
        def edit(self,crop,prompt,path,*,role):
            Image.new('RGBA',crop.size,'blue').save(path)
            return {'source_offset_xy':[0,0],'source_extent_wh':list(crop.size)}
    result=review(scene,{'node':{'fill':'#00FF00'}},[],[],source,source,tmp_path/'review',Runtime(),
        images=registry,selected=[],editor=Editor())
    asset=result['assets'][0]
    with Image.open(asset['path']) as image:
        assert image.getpixel((30,10))[3]==0  # copied original pixels cannot restore the node
        assert image.getpixel((30,22))[3]==0  # edited pixels cannot cover its native replacement
        assert image.getpixel((5,5))[3]==255  # untouched target exterior remains
    assert not result['drop_shape_keys']


def test_failed_edit_without_native_target_remains_explicitly_unresolved(tmp_path):
    source,scene,registered=fixture(tmp_path)
    scene['objects']={};scene['parts']={}
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'texture',
            'edit_target':'Keep texture on pure white','reason':'No native target'})
    class Editor:
        def edit(self,*args,**kwargs):raise ValueError('invalid canvas')
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered,editor=Editor())
    assert not result['assets'] and not result['registry_ids_consumed']
    assert result['unresolved_registry_ids']==['I0001']
    assert result['reviews'][0]['action']=='unresolved'


def test_successful_asset_reuse_requires_same_target_scope_and_bytes(tmp_path):
    source,scene,registered=fixture(tmp_path)
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'background',
            'replace_parts':['body'],'edit_target':'Keep cloud on white','reason':'wrong contour'})
    class Editor:
        calls=0
        def edit(self,crop,prompt,path,*,role):
            self.calls+=1;Image.new('RGBA',(80,60),(0,0,255,255)).save(path)
            return {'source_offset_xy':[4,4],'source_extent_wh':[80,60]}
    editor=Editor()
    first=review(scene,{},[],[],source,source,tmp_path/'first',Runtime(),images=registered,editor=editor)
    second=review(scene,{},[],[],source,source,tmp_path/'second',Runtime(),images=registered,editor=editor,cached_assets=first['assets'])
    assert editor.calls==1 and second['reviews'][0]['reused_validated_asset']
    changed=[{**first['assets'][0],'replace_parts':['different_owner_part']}]
    review(scene,{},[],[],source,source,tmp_path/'third',Runtime(),images=registered,editor=editor,cached_assets=changed)
    assert editor.calls==2
    class ChangedRequest(Runtime):
        def call(self,*args):
            result=super().call(*args);result['remove_overlays']=['newly identified external line'];return result
    review(scene,{},[],[],source,source,tmp_path/'fourth',ChangedRequest(),images=registered,editor=editor,cached_assets=first['assets'])
    assert editor.calls==3  # A corrected edit instruction must not reuse stale pixels.


def test_source_texture_replaces_all_owned_grid_strokes_in_one_commit(tmp_path):
    source,scene,_=fixture(tmp_path)
    scene['objects'].pop('network');scene['objects']['cloud']['candidate']['label']='mosaic image plane'
    scene['parts']['bad_grid']={'id':'bad_grid','object_id':'cloud','kind':'path','status':'ok',
        'bbox_2d':[15,25,100,25],'commands':[['M',15,25],['L',100,25]]}
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'crop','role':'texture',
            'replace_parts':['body','bad_grid'],'reason':'wrong grid density'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),selected=['cloud'])
    assert len(result['assets'])==1 and set(result['drop_shape_keys'])=={'body','bad_grid'}
    assert result['assets'][0]['edit']['requests']==0


def test_failed_reedit_preserves_verified_image_only_target_but_not_changed_text_ownership(tmp_path):
    source,scene,registered=fixture(tmp_path);scene['objects']={};scene['parts']={}
    class Runtime:
        target='blue cloud'
        def call(self,*args):return args[5]({'action':'image','method':'edit','role':'background',
            'replace_parts':[],'edit_target':self.target,'reason':'clean foreground'})
    class Editor:
        fail=False
        def edit(self,crop,prompt,path,*,role):
            if self.fail:raise ValueError('Edited object reaches padded canvas edge')
            Image.new('RGBA',crop.size,'blue').save(path)
            return {'source_offset_xy':[0,0],'source_extent_wh':list(crop.size)}
    runtime=Runtime();editor=Editor()
    first=review(scene,{},[],[],source,source,tmp_path/'first',runtime,images=registered,editor=editor)
    runtime.target='blue cloud silhouette';editor.fail=True
    second=review(scene,{},[],[],source,source,tmp_path/'second',runtime,images=registered,editor=editor,cached_assets=first['assets'])
    assert second['assets']==first['assets'] and second['reviews'][0]['fallback_previous_asset']
    third=review(scene,{},[],[{'text':'A','bbox_2d':[30,30,40,40]}],source,source,tmp_path/'third',runtime,
        images=registered,editor=editor,cached_assets=first['assets'])
    assert not third['assets'] and third['reviews'][0]['action']=='unresolved'


def test_crop_cannot_bake_a_later_native_text_box_into_the_image(tmp_path):
    source,scene,_=fixture(tmp_path)
    class Runtime:
        def call(self,*args):return args[5]({'action':'image','method':'crop','role':'texture',
            'replace_parts':['body'],'edit_target':'keep texture','reason':'wrong texture'})
    result=review(scene,{},[],[{'bbox_2d':[40,30,50,40]}],source,source,tmp_path/'review',Runtime(),selected=['cloud'])
    assert not result['assets'] and not result['drop_shape_keys']
    assert 'separate native text' in result['reviews'][0]['reason']
    assert 'asset_error' not in result['reviews'][0]  # rejected by the decision validator, before extraction


def test_two_raster_assets_cannot_replace_the_same_owner(tmp_path):
    p=Presentation();p.slide_width=p.slide_height=1000000;s=p.slides.add_slide(p.slide_layouts[6])
    s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,500000,500000).name='owner';p.save(tmp_path/'native.pptx')
    image=tmp_path/'img.png';Image.new('RGBA',(20,20),'blue').save(image)
    a={'id':'new','owner':'owner','role':'texture','path':str(image),'sha256':sha(image),'placement_source_half_open':[0,0,50,50]}
    with pytest.raises(ValueError,match='Two assets'):
        commit(tmp_path/'native.pptx',tmp_path/'result.pptx',[a,{**a,'id':'old'}],[100,100])
    assert not (tmp_path/'result.pptx').exists()


def test_uncontested_registered_icon_survives_without_a_new_agent_or_editor(tmp_path):
    source,scene,registered=fixture(tmp_path)
    scene['objects']={};scene['parts']={};registered[0]['category']='complex_icon'
    crop=tmp_path/'old_icon.png';Image.open(source).crop((10,10,105,90)).save(crop)
    registered[0].update(path=str(crop),sha256=sha(crop),extraction_method='source_crop',
        placement_source_half_open=[10,10,105,90])
    class Forbidden:
        def call(self,*args,**kwargs):raise AssertionError('No representation conflict to review')
        def edit(self,*args,**kwargs):raise AssertionError('Do not regenerate a working registered icon')
    result=review(scene,{},[],[],source,source,tmp_path/'review',Forbidden(),images=registered,editor=Forbidden())
    assert result['model_calls']==0 and len(result['assets'])==1
    asset=result['assets'][0]
    assert asset['placement_source_half_open']==[10,10,105,90]
    assert Image.open(asset['path']).getpixel((35,25))[:3]==Image.open(crop).getpixel((35,25))[:3]


def test_matched_delegated_image_without_native_parts_is_realized(tmp_path):
    source,scene,registered=fixture(tmp_path)
    scene['parts']={}
    scene['objects']={'owner':scene['objects']['cloud']}
    scene['objects']['owner'].update(part_keys=[],state='unresolved',declared_ownership={
        'retained_roles':{},'delegated_parts':[{'owner':'image','id':registered[0]['id']}]})
    matched,unmatched=owner_matches(scene,registered)
    assert matched=={'owner':registered} and not unmatched
    class Runtime:
        def call(self,*args):
            return args[5]({'action':'image','method':'crop','role':'icon',
                'replace_parts':[],'reason':'Registered illustration owns this complete target'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=registered)
    assert result['model_calls']==1 and len(result['assets'])==1
    assert result['assets'][0]['owner']=='owner'
    assert result['drop_shape_keys']==[]


def test_scoped_symbols_are_above_all_retained_cell_fills(tmp_path):
    deck=Presentation();deck.slide_width=1000000;deck.slide_height=1000000
    slide=deck.slides.add_slide(deck.slide_layouts[6])
    for name in ['cell_a','glyph_a','cell_b','glyph_b','foreground']:
        slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,100000,100000).name='owner/'+name
    native=tmp_path/'native.pptx';deck.save(native)
    image=tmp_path/'symbols.png';Image.new('RGBA',(10,10),(0,0,255,255)).save(image)
    asset={'id':'symbols','owner':'owner','role':'icon','path':str(image),'sha256':sha(image),
           'placement_source_half_open':[0,0,10,10],
           'replace_names':['owner/glyph_a','owner/glyph_b'],'replace_parts':[],
           'edit':{'preserved_native_frames':['cell_a','cell_b']}}
    output=tmp_path/'output.pptx';commit(native,output,[asset],[100,100])
    assert [s.name for s in Presentation(output).slides[0].shapes]==[
        'owner/cell_a','owner/cell_b','IMG symbols','owner/foreground']


def test_registered_inner_icon_never_replaces_enclosing_native_card(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(800,600),'white')
    ImageDraw.Draw(im).rectangle((40,40,60,60),fill='blue');im.save(source)
    def part(name,b):
        x,y,r,t=b
        return {'id':name,'object_id':'card','status':'ok','kind':'path','bbox_2d':b,
                'commands':[['M',x,y],['L',r,y],['L',r,t],['L',x,t],['Z']]}
    scene={'source_size':[800,600],'source_sha256':sha(source),
        'objects':{'card':{'candidate':{'label':'stacked robot icon','kind':'compound_symbol','bbox_2d':[10,10,100,100]}}},
        'parts':{'background':part('background',[10,10,100,100]),'symbol':part('symbol',[40,40,60,60])}}
    reg=[{'id':'I1','target':'robot symbol','category':'complex_icon','source_box_half_open':[38,38,62,62]}]
    class Runtime:
        calls=0
        def call(self,*args):
            self.calls+=1
            assert [p['key'] for p in args[3]['parts']]==['P0']
            assert args[3]['parts'][0]['bbox']==[40,40,60,60]
            return args[5]({'action':'image','method':'crop','role':'icon','replace_parts':['symbol'],'reason':'source symbol'})
    runtime=Runtime();result=review(scene,{},[],[],source,source,tmp_path/'review',runtime,images=reg)
    assert runtime.calls==1 and len(result['assets'])==1
    assert result['drop_shape_keys']==['symbol'] and result['registry_ids_consumed']==['I1']
    deck=Presentation();slide=deck.slides.add_slide(deck.slide_layouts[6])
    deck.slide_width=8000000;deck.slide_height=6000000
    group=slide.shapes.add_group_shape()
    for name in ['background','symbol']:
        group.shapes.add_shape(MSO_SHAPE.RECTANGLE,100000,100000,900000,900000).name='card/'+name+' [unresolved]'
    group.left=200000;group.width=1800000
    original_background=group.shapes[0]._element.xml
    deck.save(tmp_path/'native.pptx')
    commit(tmp_path/'native.pptx',tmp_path/'final.pptx',result['assets'],[800,600])
    children=Presentation(tmp_path/'final.pptx').slides[0].shapes[0].shapes
    names=[s.name for s in children]
    assert 'card/background [unresolved]' in names and 'card/symbol [unresolved]' not in names and names.count('IMG U_card')==1
    assert children[0]._element.xml==original_background
    pic=next(s for s in children if s.name=='IMG U_card')
    # Parent scale 2 and translated origin; the image still lands at source x.
    assert abs(200000+(pic.left-100000)*2-result['assets'][0]['placement_source_half_open'][0]*10000)<=2


def test_small_registered_illustration_replaces_subowners_in_one_task(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(800,600),'white')
    ImageDraw.Draw(im).rectangle((30,30,70,50),fill='blue');im.save(source)
    parts={name:{'object_id':name,'kind':'rectangle','status':'ok','bbox_2d':b,
            'commands':[['M',b[0],b[1]],['L',b[2],b[1]],['L',b[2],b[3]],['L',b[0],b[3]],['Z']]}
           for name,b in [('icon',[30,30,45,45]),('ball',[60,30,70,40])]}
    scene={'source_size':[800,600],'source_sha256':sha(source),'parts':parts,
        'objects':{k:{'candidate':{'kind':'compound_symbol' if k=='icon' else 'ellipse',
                   'label':'small icon' if k=='icon' else 'ball','bbox_2d':p['bbox_2d']}}
                   for k,p in parts.items()}}
    registry=[{'id':'I1','category':'illustration','target':'complete small symbol','source_box_half_open':[20,20,90,60]}]
    class Runtime:
        calls=0
        def call(self,*args):
            self.calls+=1
            assert {p['key'] for p in args[3]['parts']}=={'P0','P1'}
            assert {tuple(p['bbox']) for p in args[3]['parts']}=={(30,30,45,45),(60,30,70,40)}
            return args[5]({'action':'image','method':'crop','role':'icon','replace_parts':['icon','ball'],'reason':'missing source detail'})
    runtime=Runtime();result=review(scene,{},[],[],source,source,tmp_path/'review',runtime,images=registry)
    assert runtime.calls==1 and len(result['assets'])==1
    assert set(result['assets'][0]['replace_names'])=={'icon/icon','ball/ball'}


def test_matched_image_replaces_contained_subowner_without_expanding_crop(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(200,200),'white')
    draw=ImageDraw.Draw(im);draw.rectangle((90,30,145,90),fill='blue')
    draw.text((100,65),'LOGO',fill='white');im.save(source)
    def part(owner,name,b,kind='path'):
        x,y,r,t=b
        return {'id':name,'object_id':owner,'kind':kind,'status':'ok','bbox_2d':b,
                'commands':[['M',x,y],['L',r,y],['L',r,t],['L',x,t],['Z']]}
    parts={'border':part('card','border',[20,20,170,100],'rectangle'),
           'texture':part('card','texture',[30,40,70,85]),
           'badge':part('symbol','badge',[90,30,145,90],'rectangle'),
           'signal':part('symbol','signal',[105,40,135,50]),
           'outside':part('neighbor','outside',[150,90,190,120],'rectangle')}
    parts['border']['native_parameters_original']={'center':[95,60],'width':150,'height':80}
    parts['badge']['native_parameters_original']={'center':[117.5,60],'width':55,'height':60}
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':parts,
           'objects':{oid:{'candidate':{'kind':'rectangle','bbox_2d':b}}
                      for oid,b in [('card',[20,20,170,100]),('symbol',[90,30,145,90]),
                                    ('neighbor',[150,90,190,120])]}}
    registry=[{'id':'I1','category':'complex_icon','target':'textured card',
               'source_box_half_open':[20,20,170,100]}]
    class Runtime:
        calls=0
        def call(self,root,tag,system,prompt,images,validate,budget):
            self.calls+=1
            offered={tuple(p['bbox']) for p in prompt['parts']}
            assert (90,30,145,90) in offered and (105,40,135,50) in offered
            assert (150,90,190,120) not in offered
            return validate({'action':'image','method':'crop','role':'icon',
                'replace_parts':['texture','badge','signal'],'replace_arrows':[],
                'reason':'The registered card contains the full textured visual unit.'})
    runtime=Runtime()
    result=review(scene,{},[],[],source,source,tmp_path/'review',runtime,
                  images=registry,selected=['card'])
    assert runtime.calls==1 and len(result['assets'])==1
    asset=result['assets'][0]
    assert set(asset['replace_names'])=={'card/texture','symbol/badge','symbol/signal'}
    crop=asset['source_box_half_open']
    assert 20<=crop[0]<=90 and 20<=crop[1]<=30
    assert 145<=crop[2]<=170 and 90<=crop[3]<=100
    deck=Presentation();deck.slide_width=2000000;deck.slide_height=2000000
    slide=deck.slides.add_slide(deck.slide_layouts[6])
    for p in parts.values():
        shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,100000,100000,500000,500000)
        shape.name=p['object_id']+'/'+p['id']
    preserved={s.name:s._element.xml for s in slide.shapes if s.name in ('card/border','neighbor/outside')}
    deck.save(tmp_path/'native.pptx')
    commit(tmp_path/'native.pptx',tmp_path/'result.pptx',[asset],[200,200])
    final=Presentation(tmp_path/'result.pptx').slides[0].shapes
    assert {s.name for s in final}=={'card/border','neighbor/outside','IMG U_card'}
    assert all(s._element.xml==preserved[s.name] for s in final if s.name in preserved)


def test_image_owns_only_its_internal_arrow_not_a_crossing_connector(tmp_path):
    source,scene,registered=fixture(tmp_path)
    arrows=[{'id':ident,'start':start,'end':end,'heads':['none','triangle'],
             'style':{'color':'#000000','width_px':2}} for ident,start,end in
            [('inner',[30,30],[70,30]),('crossing',[0,40],[140,40])]]
    class Runtime:
        def call(self,*args):
            assert [a['id'] for a in args[3]['contained_arrow_candidates']]==['inner']
            return args[5]({'action':'image','method':'crop','role':'icon','replace_parts':['body'],'replace_arrows':['inner'],'reason':'intrinsic symbol'})
    result=review(scene,{},arrows,[],source,source,tmp_path/'review',Runtime(),images=registered)
    assert result['drop_arrow_names']==['inner'] and result['assets'][0]['replace_arrows']==['inner']


def test_edited_background_preserves_source_slide_clip_without_rescaling(tmp_path):
    p=tmp_path/'image.png';im=Image.new('RGBA',(40,20),'blue');im.putpixel((30,10),(255,0,0,255));im.save(p)
    b=clip_source_edge(p,[80,20,120,40],[100,60],[78,18,100,42])
    assert b==[80,20,100,40] and Image.open(p).size==(20,20)
    assert Image.open(p).getpixel((10,10))==(0,0,255,255)
    with pytest.raises(ValueError):clip_source_edge(p,[80,20,120,40],[100,60],[78,18,98,42])
