from pathlib import Path
import pytest

from full_recovery.external_arrow_products import bind
from full_recovery.prepare import save, read
from test_external_arrow_products import inputs


def test_only_explicit_missing_arrow_cases_may_be_deferred(tmp_path):
    fresh,root,case,item,manifest,record=inputs(tmp_path)
    pending={**item,'id':'pending'}
    with pytest.raises(ValueError,match='product missing'):
        bind(root,[item,pending],fresh/'unapproved.json')
    bound=bind(root,[item,pending],fresh/'binding.json',pending_ids=['pending'])
    assert [r['id'] for r in bound['records']]==[item['id']]
    assert [r['id'] for r in bound['pending']]==['pending']
    # A pending nomination is not permission to skip an available input.
    bound=bind(root,[item],fresh/'available.json',pending_ids=[item['id']])
    assert len(bound['records'])==1 and not bound['pending']
    with pytest.raises(ValueError,match='requested source cases'):
        bind(root,[item],fresh/'unknown.json',pending_ids=['not_requested'])


def test_pending_arrow_keeps_shape_and_text_work_but_defers_only_composition(tmp_path,monkeypatch):
    from full_recovery import fresh_pipeline,merge_existing
    fresh,root,case,item,manifest,record=inputs(tmp_path)
    pending={**item,'id':'pending'}
    bind(root,[item,pending],fresh/'external_arrow_inputs.json',pending_ids=['pending'])
    settings={'workflow_scope':'fresh_full','arrow_driver':'existing_products',
        'external_arrow_inputs':str(fresh/'external_arrow_inputs.json'),'fresh_arrow_root':str(root),
        'source_project':str(tmp_path),'text_workers':1,'glm_url':'http://unused.invalid',
        'text_ocr_python':'python','text_models':'unused','text_recognition_mode':'staged',
        'text_ocr_mode':'geometry','text_ocr_template_validation':'on','text_threshold':.8,
        'text_max_score_drop':.05,'text_ocr_max_side':1536,'text_crop_refine':True}
    save(fresh/'settings.json',settings);save(fresh/'source_manifest.json',{'images':[item,pending]})
    (fresh/'logs').mkdir();calls=[]
    class Child:
        returncode=0
        def __init__(self,argv,**kwargs):calls.append(argv)
        def poll(self):return 0
    monkeypatch.setenv('STYLE_CLUSTER_WORKER','1')
    monkeypatch.setattr(fresh_pipeline.subprocess,'Popen',Child)
    monkeypatch.setattr(fresh_pipeline.signal,'signal',lambda *args:None)
    def inspect(*args):
        assert args[-1]==[item['id']]
        return [record],[]
    monkeypatch.setattr(merge_existing,'inspect_sources',inspect)
    def compose(run,*,record_stream):
        save(run/'shape/geometry/cases'/item['id']/'complete.json',{})
        assert list(record_stream)==[record]
        save(run/'summary.json',{'exported':1})
    monkeypatch.setattr(merge_existing,'execute',compose)
    fresh_pipeline.execute(fresh)
    assert [Path(argv[3]).name for argv in calls].count('run_shapes.py')==1
    assert [Path(argv[3]).name for argv in calls].count('text_stage.py')==2
    summary=read(fresh/'summary.json')
    assert summary['state']=='completed_with_pending_arrow_inputs'
    assert summary['pending_arrow_ids']==['pending'] and summary['shape_text_requested']==2
    assert read(fresh/'sources.json')['missing'][0]['id']=='pending'
