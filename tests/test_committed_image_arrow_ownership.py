from pathlib import Path
from unittest.mock import patch

from PIL import Image

from full_recovery.image_realization import realize
from full_recovery.handoff import reconcile


def test_successful_image_commits_its_exact_internal_arrows_without_model_review(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    texture={'assets':[{'id':'image','pending_edit':False,'replace_arrows':['internal'],
                        'replace_parts':[]}], 'drop_arrow_names':[], 'drop_shape_keys':[]}
    realize(texture,{'parts':{}},{},source,{},None)
    assert texture['drop_arrow_names']==['internal']
    scene={'parts':{}}
    arrows={'arrows':[{'id':ident,'start':[10,10],'end':[20,10],'heads':['none','none']}
                      for ident in ['internal','external']]}
    # A successful earlier line choice must not be asked to repeat itself.
    plan=reconcile(scene,{'parts':{}},arrows,{'texts':[]},quality={
        'ordinary_line_owner':'local_competitor_groups','line_ownership_complete':True},texture=texture)
    assert plan['drop_arrow_names']==['internal']


def test_failed_asset_does_not_commit_its_proposed_arrow_removal(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    asset={'id':'failed','owner':'owner','pending_edit':True,'replace_arrows':['external'],
           'replace_parts':[],'registry_ids':[],'path':str(tmp_path/'image.png'),
           'edit_context_half_open':[0,0,100,100],
           'edit_request':{'removal_paths':[]},'edit':{},'native_enclosure':None}
    texture={'assets':[asset],'drop_arrow_names':[],'drop_shape_keys':[],
             'drop_shape_names':[],'registry_ids_consumed':[]}
    realize(texture,{'parts':{}},{},source,{},None)
    assert texture['assets']==[] and texture['drop_arrow_names']==[]
    assert texture['realization_failures'][0]['fallback']=='retain_native_owner'


def test_already_realized_assets_replay_same_commit_set_and_keep_crossing_arrows(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    texture={'assets':[{'id':'one','pending_edit':False,'replace_parts':[],'replace_arrows':['inside']},
                       {'id':'two','replace_parts':[],'replace_arrows':[]}],
             'drop_arrow_names':['stale_proposal'], 'drop_shape_keys':[]}
    for _ in range(2):
        realize(texture,{'parts':{}},{},source,{},None)
        assert texture['drop_arrow_names']==['inside']
