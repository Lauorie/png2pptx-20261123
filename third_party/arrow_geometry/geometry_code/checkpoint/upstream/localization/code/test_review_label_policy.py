"""A replacement region without a label takes its patch reason instead of failing."""
import copy
import threading

from PIL import Image

import local_review
from local_review import validate
from review_label_policy import fill


def row(key, box=None, label='visible target'):
    return dict(id=key, bbox=box or [10, 10, 30, 30], render_type='linework',
                grouping='single', label=label, extent='complete')


def answer(refs, regions, reason='two dashed red arrows of one fan-out'):
    return dict(frame='LOCAL', decision='edit', keep_ids=[], discard=[], inspect=[],
                replacements=[dict(source_ids=refs, regions=regions, reason=reason)])


def test_a_missing_label_takes_the_patch_reason():
    child = row('R1'); child.pop('label')
    filled, audit = fill(answer(['C1'], [child]))
    assert filled['replacements'][0]['regions'][0]['label'] == 'two dashed red arrows of one fan-out'
    assert audit['filled'] == ['R1']


def test_a_blank_label_is_also_filled():
    filled, _ = fill(answer(['C1'], [row('R1', label='  ')]))
    assert filled['replacements'][0]['regions'][0]['label'] == 'two dashed red arrows of one fan-out'


def test_a_label_the_model_wrote_is_kept():
    filled, audit = fill(answer(['C1'], [row('R1', label='grey shaft')]))
    assert filled['replacements'][0]['regions'][0]['label'] == 'grey shaft'
    assert audit['filled'] == []


def test_the_raw_answer_is_not_mutated():
    child = row('R1'); child.pop('label')
    raw = answer(['C1'], [child]); saved = copy.deepcopy(raw)
    fill(raw)
    assert raw == saved


def test_malformed_answers_are_left_for_the_validator():
    for raw in ({'decision': 'expand', 'reason': 'more'}, {'replacements': 'x'}, {'replacements': [3]},
                {'replacements': [{'regions': None, 'reason': 'r'}]}, None):
        assert fill(raw)[0] == raw


def test_validation_accepts_an_unlabelled_replacement_and_returns_it_unchanged():
    child = row('R1'); child.pop('label')
    raw = answer(['C1'], [child]); saved = copy.deepcopy(raw)
    assert validate(raw, (100, 100), [row('C1')], True) == saved


def test_validation_still_rejects_a_replacement_with_neither_label_nor_reason():
    child = row('R1'); child.pop('label')
    raw = answer(['C1'], [child], reason='')
    try:
        validate(raw, (100, 100), [row('C1')], False)
    except ValueError:
        return
    raise AssertionError('an unlabelled, unexplained replacement must be rejected')


class _SplitModel:
    """Fails the whole batch, then answers the two halves only while both are open."""

    def __init__(self):
        self.together = threading.Barrier(2, timeout=5)

    def ask(self, folder, prompt, images, validator, system=None):
        block = folder.parent.parent.name
        if block == 'B001':
            raise ValueError('whole batch refused')
        self.together.wait()
        for kept in ('C1', 'C2'):
            reply = dict(frame='LOCAL', decision='edit', keep_ids=[kept], replacements=[], discard=[], inspect=[])
            try:
                return validator(reply)
            except ValueError:
                continue
        raise AssertionError('each half holds exactly one candidate')


def test_the_two_halves_of_a_failed_batch_are_reviewed_at_the_same_time(tmp_path, monkeypatch):
    monkeypatch.setattr(local_review, 'latest', lambda *args: None)
    source = tmp_path / 'inputs' / 'case'
    source.mkdir(parents=True)
    Image.new('RGB', (400, 300), 'white').save(source / 'source.png')
    rows = [row('C1', [100, 100, 200, 160]), row('C2', [120, 110, 220, 170])]
    current, issues = local_review.run(tmp_path, 'case', rows, _SplitModel(),
                                       {'local_review_rounds': 1}, lambda stage: None)
    assert issues == []
    assert sorted(r['id'] for r in current) == ['C1', 'C2']
