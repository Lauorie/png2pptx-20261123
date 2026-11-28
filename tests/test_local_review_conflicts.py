"""One disputed SAM claim in a region review must cost that claim, not the whole review."""
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor'), str(ROOT / 'experiment/shape_sam_guided')]

import pytest

from shape_integration.common import save
from shape_integration.sam_fixed import FixedSamEvidence
from shape_integration.shape import local_review_checker

CANDIDATES = [{'id': 'G001', 'bbox_2d': [0, 0, 100, 100]}]


def _evidence(folder):
    save(folder/'sam_evidence.json', {'shown': [{'id': 'S1', 'bbox_original': [40, 40, 50, 50]},
                                                {'id': 'S2', 'bbox_original': [0, 0, 100, 100]}]})
    return object.__new__(FixedSamEvidence)


def _reply():
    return {'reviews': [{'id': 'G001', 'action': 'keep', 'bbox_2d': None, 'reason': 'the whole grid'}],
            'new_objects': [{'id': 'N001', 'bbox_2d': [60, 20, 90, 70]}], 'uncertain': [], 'more_pending': False,
            'sam_decisions': [{'sam_id': 'S1', 'action': 'covered', 'target': 'G001', 'reason': 'one cell of the grid'},
                              {'sam_id': 'S2', 'action': 'covered', 'target': 'G001', 'reason': 'the grid itself'}]}


def test_a_part_claimed_as_covered_by_its_compound_keeps_the_rest_of_the_review(tmp_path):
    raw = _reply()
    original = deepcopy(raw)
    result = local_review_checker(_evidence(tmp_path), lambda reply: reply, tmp_path, CANDIDATES)(raw)
    assert result['reviews'] == original['reviews'] and result['new_objects'] == original['new_objects']
    assert [row['action'] for row in result['sam_decisions']] == ['uncertain', 'covered']
    assert raw == original


def test_the_disputed_part_stays_open_for_the_later_coverage_review(tmp_path):
    result = local_review_checker(_evidence(tmp_path), lambda reply: reply, tmp_path, CANDIDATES)(_reply())
    assert result['more_pending'] is True
    assert any(note.startswith('S1:') for note in result['uncertain'])
    assert result['sam_coverage_conflicts'][0]['sam_id'] == 'S1'


def test_a_reply_that_skips_a_sam_id_is_still_refused(tmp_path):
    raw = _reply()
    raw['sam_decisions'] = raw['sam_decisions'][:1]
    with pytest.raises(ValueError):
        local_review_checker(_evidence(tmp_path), lambda reply: reply, tmp_path, CANDIDATES)(raw)


def test_without_sam_evidence_the_checker_is_used_as_it_is(tmp_path):
    def checker(reply):
        return reply

    assert local_review_checker(None, checker, tmp_path, CANDIDATES) is checker
