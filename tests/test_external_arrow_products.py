import json
from pathlib import Path

from PIL import Image
import pytest

from full_recovery.external_arrow_products import bind, verify
from full_recovery.fresh_pipeline import validate_sources
from full_recovery.prepare import save, sha


def inputs(tmp_path):
    original = tmp_path/'original.png'; Image.new('RGB', (24, 16), 'white').save(original)
    item = {'id': 'sample', 'original': str(original), 'source_size': [24, 16],
            'source_sha256': sha(original)}
    root = tmp_path/'selected_arrows'; case = root/'geometry/cases/sample'; case.mkdir(parents=True)
    Image.new('RGB', (24, 16), 'white').save(case/'clean.png')
    save(case/'scene.json', {'size': [24, 16], 'arrows': []})
    (case/'arrows.pptx').write_bytes(b'explicit-native-product')
    fresh = tmp_path/'new_run'; fresh.mkdir()
    manifest = bind(root, [item], fresh/'external_arrow_inputs.json')
    owned = fresh/'shape_scene.json'; save(owned, {'new': True})
    record = {'item': item, 'paths': {'shape_scene': str(owned),
        **{k: v for k, v in manifest['records'][0]['paths'].items() if k != 'arrow_clean'}},
        'sha256': {'shape_scene': sha(owned),
        **{k: v for k, v in manifest['records'][0]['sha256'].items() if k != 'arrow_clean'}}}
    return fresh, root, case, item, manifest, record


def test_only_explicit_arrow_inputs_may_enter_a_fresh_shape_text_run(tmp_path):
    fresh, root, case, item, manifest, record = inputs(tmp_path)
    validate_sources(fresh, [record], manifest)
    old_text = tmp_path/'old_text.json'; save(old_text, {'old': True})
    record['paths']['text_scene'] = str(old_text); record['sha256']['text_scene'] = sha(old_text)
    with pytest.raises(ValueError, match='Historical branch product'):
        validate_sources(fresh, [record], manifest)


def test_changed_arrow_input_cannot_silently_use_the_same_binding(tmp_path):
    fresh, root, case, item, manifest, record = inputs(tmp_path)
    (case/'arrows.pptx').write_bytes(b'updated product')
    with pytest.raises(ValueError, match='arrow input changed'):
        verify(manifest)


def test_arrow_pixels_must_match_the_current_original(tmp_path):
    fresh, root, case, item, manifest, record = inputs(tmp_path)
    image = Image.new('RGB', (24, 16), 'white'); image.putpixel((5, 5), (0, 0, 0)); image.save(case/'clean.png')
    with pytest.raises(ValueError, match='source pixels differ'):
        bind(root, [item], fresh/'wrong_binding.json')


def test_existing_arrow_policy_never_starts_an_arrow_process(tmp_path, monkeypatch):
    from full_recovery import fresh_pipeline, merge_existing
    fresh, root, case, item, manifest, record = inputs(tmp_path)
    settings = {'workflow_scope': 'fresh_full', 'arrow_driver': 'existing_products',
        'external_arrow_inputs': str(fresh/'external_arrow_inputs.json'), 'fresh_arrow_root': str(root),
        'source_project': str(tmp_path), 'text_workers': 1, 'glm_url': 'http://unused.invalid',
        'text_ocr_python': 'python', 'text_models': 'unused', 'text_recognition_mode': 'staged',
        'text_ocr_mode': 'geometry', 'text_ocr_template_validation': 'on',
        'text_threshold': .8, 'text_max_score_drop': .05, 'text_ocr_max_side': 1536,
        'text_crop_refine': True}
    save(fresh/'settings.json', settings); save(fresh/'source_manifest.json', {'images': [item]})
    (fresh/'logs').mkdir(); calls = []
    class Child:
        returncode = 0
        def __init__(self, argv, **kwargs): calls.append(argv)
        def poll(self): return 0
    monkeypatch.setenv('STYLE_CLUSTER_WORKER', '1')
    monkeypatch.setattr(fresh_pipeline.subprocess, 'Popen', Child)
    monkeypatch.setattr(fresh_pipeline.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(merge_existing, 'inspect_sources', lambda *args: ([record], []))
    def compose(run, *, record_stream):
        (run/'texts/cases'/item['id']).mkdir(parents=True)
        save(run/'texts/cases'/item['id']/'summary.json', {})
        (run/'shape/geometry/cases'/item['id']).mkdir(parents=True)
        save(run/'shape/geometry/cases'/item['id']/'complete.json', {})
        assert list(record_stream) == [record]
        save(run/'summary.json', {'exported': 1})
    monkeypatch.setattr(merge_existing, 'execute', compose)
    fresh_pipeline.execute(fresh)
    assert len(calls) == 2
    assert {Path(argv[3]).name for argv in calls} == {'run_shapes.py', 'text_stage.py'}
    summary = json.loads((fresh/'summary.json').read_text())
    assert summary['arrow_inference_enabled'] is False and summary['shape_text_branches_fresh'] is True
    assert summary['historical_products_reused'] is True
