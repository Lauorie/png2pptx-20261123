"""Copied into a new reference-text freeze; exercised by its backend suite."""
from copy import deepcopy

from PIL import Image

from shape_integration.common import read, save
from shape_integration.texts import options, two_stage_crop as pipeline
from shape_integration.texts.controller import mapped_box
from test_inventory_visual_recovery import row


def test_one_rejected_proposal_can_recover_multiple_visible_occurrences(tmp_path,monkeypatch):
    from shape_integration.texts import inventory_recovery,model
    root=tmp_path/'inventory';save(root/'complete.json',{'raw':'same-image response'})
    first={**row('', 'repeated label',[10,10,100,25]),'replaces':[]}
    rejected={**first,'bbox_2d':[10,50,100,65],'replaces':['invented_a','invented_b']}
    monkeypatch.setattr(inventory_recovery,'latest_complete',
        lambda folder:({'texts':[first,rejected]},root/'complete.json'))
    calls=[]
    def ask(runtime,folder,prompt,images,validator,opt):
        calls.append(prompt)
        assert len(prompt['accepted_source_occurrences'])==1
        assert 'return every missing occurrence' in ' '.join(prompt['rules'])
        return validator({'texts':[{**first,'bbox_2d':[10,y,100,y+15]} for y in (50,90)]})
    monkeypatch.setattr(model,'ask',ask)
    result=inventory_recovery.recover(root,[200,140],options({}),object(),'invalid repeated proposal')
    assert len(calls)==1 and len(result['texts'])==3
    assert [r['bbox_2d'][1] for r in result['texts']]==[10,50,90]


def test_recovery_keeps_original_occurrence_anchor_and_local_resolution(tmp_path, monkeypatch):
    source = tmp_path/'source.png'
    Image.new('RGB', (1000, 600), 'white').save(source)
    rows = [{'id': ident, 'text': 'same', 'kind': 'text', 'latex': '',
        'runs': [{'text': 'same', 'script': 'normal', 'color': '#000000'}],
        'bbox_2d': [400, y, 430, y+20], 'font': 'Times New Roman', 'color': '#000000',
        'bold': False, 'italic': False, 'underline': False, 'rotation_deg': 0,
        'legibility': 'clear', 'replaces': [], 'font_size_px': 18,
        'frame_px': [400, y, 30, 20], 'resolved_font': 'Times New Roman'}
        for ident, y in [('upper', 250), ('lower', 290)]]
    baseline = tmp_path/'baseline'
    save(baseline/'scene.json', {'texts': rows})
    Image.new('RGB', (1000, 600), 'white').save(baseline/'native.png')
    output = tmp_path/'result'; output.mkdir()
    recovered = []
    class Runtime:
        def for_stage(self, **kwargs):
            return self
        def call(self, folder, tag, system, prompt, images, validator, budget):
            frame = read(folder.parent/'input/frame.json')
            if prompt['stage'] == '1_bbox_only':
                if 'recovery' in folder.parts:
                    assert frame['box'] != [0, 0, 1000, 600]
                    assert max(frame['size']) >= 1000
                    for target in prompt['targets']:
                        anchor = mapped_box(target['rough_bbox_2d'], frame, inverse=True)
                        expected = next(r['bbox_2d'] for r in rows if r['id'] == target['id'])
                        assert all(abs(a-b) < 1e-6 for a, b in zip(anchor, expected))
                    recovered.extend(t['id'] for t in prompt['targets'])
                    return validator({'items': [{'id': t['id'], 'status': 'ambiguous', 'units': [],
                        'reason': 'Repeated occurrences remain ambiguous'} for t in prompt['targets']]})
                # A first reply assigns both IDs to the same incorrect occurrence.
                return validator({'items': [{'id': t['id'], 'status': 'resolved',
                    'units': [{'bbox_2d': [20, 20, 120, 60]}]} for t in prompt['targets']]})
            values = []
            for target in prompt['targets']:
                unit = deepcopy(next(r for r in rows if r['id'] == target['id']))
                unit.pop('id')
                unit.update(direction_status='confirmed', visual_line_count=1, reason='Visible local label')
                values.append({'id': target['id'], 'status': 'resolved', 'units': [unit]})
            return validator({'items': values})
    monkeypatch.setattr(pipeline.layout, 'fit_all', lambda rows, *args, **kwargs: (rows, []))
    monkeypatch.setattr(pipeline.export, 'final', lambda rows, source, folder, *args:
        Image.new('RGB', (1000, 600), 'white').save(folder/'native.png'))
    summary = pipeline.finish(source, output, baseline, options({'max_tokens': 1000}), Runtime())
    assert set(recovered) == {'upper', 'lower'}
    final = read(output/'final/scene.json')['texts']
    assert [(r['id'], r['bbox_2d']) for r in final] == [(r['id'], r['bbox_2d']) for r in rows]
    assert set(summary['unresolved_ids']) == {'upper', 'lower'}


def test_unseen_far_target_cannot_return_an_outside_view_coordinate(tmp_path, monkeypatch):
    source = tmp_path/'source.png'
    Image.new('RGB', (1000, 400), 'white').save(source)
    target = row('target', 'fallible text', [700, 100, 760, 120])
    baseline = tmp_path/'baseline'
    save(baseline/'scene.json', {'texts': [target]})
    Image.new('RGB', (1000, 400), 'white').save(baseline/'native.png')
    output = tmp_path/'result'; output.mkdir()
    attempted_frames = []
    class Runtime:
        def for_stage(self, **kwargs):
            return self
        def call(self, folder, tag, system, prompt, images, validator, budget):
            f = read(folder.parent/'input/frame.json')
            attempted_frames.append(f)
            assert f['box'][0] > 160
            # A valid source coordinate can still be outside this particular view.
            return validator({'items': [{'id': 'target', 'status': 'resolved',
                'units': [{'bbox_2d': mapped_box([100, 100, 160, 120], f)}]}]})
    monkeypatch.setattr(pipeline.layout, 'fit_all', lambda rows, *args, **kwargs: (rows, []))
    monkeypatch.setattr(pipeline.export, 'final', lambda rows, source, folder, *args:
        Image.new('RGB', (1000, 400), 'white').save(folder/'native.png'))
    summary = pipeline.finish(source, output, baseline, options({'max_tokens': 1000}), Runtime())
    selected = read(output/'final/scene.json')['texts']
    assert attempted_frames and selected[0]['bbox_2d'] == target['bbox_2d']
    assert summary['unresolved_ids'] == ['target']
    assert not summary['accepted_ids']
