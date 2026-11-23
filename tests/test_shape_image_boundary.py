"""Source ownership and bounded progress must agree across existing stages."""
from copy import deepcopy
from pathlib import Path
import unittest
import pytest
from PIL import Image, ImageDraw

from shape_integration.common import read, save, sha
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.contracts import normalize_observation
from shape_integration.fitting.critic import validate_review
from test_declared_ownership import observation
import test_pipeline as fixtures
from test_pipeline import FakeRuntime
from test_shape_robustness import blocks


def test_observer_sees_registered_image_extent_without_modifying_clean_pixels(tmp_path):
    candidate={'id':'outer','kind':'rectangle','label':'outer container','bbox_2d':[10,10,180,100]}
    save(tmp_path/'image_assets.json',[{'id':'I1','bbox_2d':[50,25,90,70]}])
    job=prepare_candidate(Image.new('RGB',(200,120),'white'),candidate,[200,120],tmp_path/'fit')
    assert job['image_asset_ids']==['I1']
    assert read(tmp_path/'fit/observer_assets.json')['assets'][0]['bbox_original']==[50,25,90,70]
    assert sha(tmp_path/'fit/clean.png')!=sha(tmp_path/'fit/overlay.png')
    with Image.open(tmp_path/'fit/clean.png') as clean:
        assert clean.getextrema()==((255,255),)*3
    assert 'image_ownership' in read(tmp_path/'fit/prompt.json')['prompt']


def test_raster_delegation_does_not_remove_an_independent_frame():
    raw=observation();raw['ownership']['parts'][1]['role']='image_content';before=deepcopy(raw);audit={}
    fit=normalize_observation(raw,'unit',[100,100],audit=audit,ownership_required=True)
    assert [p['id'] for p in fit['parts']]==['node']
    assert audit['declared_ownership']['delegated_parts']==[{'id':'route','role':'image_content','owner':'image'}]
    assert raw==before


def test_revision_assessment_requires_source_evidence_and_explicit_outcome():
    review={'decision':'revise','issues':[{'kind':'geometry','parts':['F1'],
        'evidence':'The old bottom edge still lies below the source.','action':'Raise only the bottom edge.'}]}
    with pytest.raises(ValueError):validate_review(review,['F1'],revision=True)
    review['revision_effect']={'outcome':'improved','evidence':'The right edge now matches the source; the old bottom offset is unchanged.'}
    assert validate_review(review,['F1'],revision=True)==review


class TestRetainedImprovement(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object

    def check_progress(self,outcome,revision):
        self.options.update(max_revisions=1,source_frame_refinement=False)
        def respond(tag,prompt,count):
            if tag=='observe':
                raw=blocks()
                if 'revision_task' in prompt:
                    for p in raw['points']:
                        if p['id'].startswith('cell7'):p['xy'][0]+=10
                return raw
            issue={'kind':'geometry','parts':['F2'],'evidence':'The bottom offset remains.','action':'Raise the bottom edge.'}
            result={'decision':'revise','issues':[issue]}
            if 'revision_effect' in prompt['output_schema']['properties']:
                result['revision_effect']={'outcome':outcome,'evidence':'Compared the changed edge and remaining bottom offset with source.'}
            return result
        record=self.run_object(FakeRuntime(self.config,respond))
        assert record['state']=='unresolved'
        assert record['stop_reason']=='revision_budget_exhausted'
        assert record['selected_revision']==revision

    def test_remaining_old_defect_cannot_erase_confirmed_progress(self):
        self.check_progress('improved','revision_01')

    def test_new_regression_keeps_previous_geometry(self):
        self.check_progress('regressed','revision_00')

    def test_uncertain_progress_keeps_previous_geometry(self):
        self.check_progress('uncertain','revision_00')


def test_deferred_image_cleanup_removes_only_measured_frame_copy(tmp_path):
    from full_recovery.image_realization import realize
    from full_recovery.image_assets import remove_frame_copies
    source=tmp_path/'source.png';image=Image.new('RGB',(200,150),'white');draw=ImageDraw.Draw(image)
    draw.rectangle([30,30,170,120],outline='black',width=2)
    draw.rectangle([65,50,105,95],fill='blue');image.save(source)
    frame={'bbox_2d':[32,32,168,118]}
    scene={'source_size':[200,150],'parts':{'frame':frame}}
    path=tmp_path/'image.png';image.convert('RGBA').save(path)
    asset={'id':'icon','role':'icon','path':str(path),'pending_edit':True,'registry_ids':['I1'],
        'edit_context_half_open':[20,20,180,130],'replace_parts':[],
        'edit_request':{'source_pixels_policy':'explicit_cleanup_masks.1','removal_paths':[],'native_labels':[]},
        'edit':{'owned_box':[20,20,180,130],'preserved_native_frames':['frame'],'separate_foreground_parts':[]}}
    texture={'assets':[asset]};realize(texture,scene,{},source,{},None)
    with Image.open(path) as result:
        assert result.getpixel((10,10))[3]==0
        assert result.getpixel((50,40))==(0,0,255,255)
    assert asset['edit']['removed_retained_frame_copies'][0]['native_frame']=='frame'
    _,audit=remove_frame_copies(image.convert('RGBA'),[0,0,200,150],image,scene,['frame'],['frame'])
    assert audit==[]
