"""A coverage repair asked to add a part may not throw away what was already fitted."""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest.mock import patch

import test_pipeline as fixtures
from shape_integration.common import read, save


def _fit(*boxes: List[float]) -> Dict[str, Any]:
    return {'parts': [{'id': str(i), 'status': 'ok', 'bbox_2d': list(box)} for i, box in enumerate(boxes)]}


MISSING = [{'kind': 'missing', 'evidence': 'top bar absent', 'action': 'add it'}]


def test_a_repair_that_keeps_only_a_corner_of_the_old_fit_discards_it() -> None:
    from shape_integration.fitting.repair_guard import discards_base
    assert discards_base(MISSING, _fit([0, 0, 100, 100]), _fit([60, 0, 100, 40]))


def test_a_repair_that_covers_the_old_fit_and_more_is_kept() -> None:
    from shape_integration.fitting.repair_guard import discards_base
    assert not discards_base(MISSING, _fit([0, 20, 100, 100]), _fit([0, 0, 100, 100], [10, 30, 40, 60]))


def test_failed_parts_do_not_count_as_fitted() -> None:
    from shape_integration.fitting.repair_guard import discards_base
    repair = {'parts': [{'id': 'a', 'status': 'ok', 'bbox_2d': [60, 0, 100, 40]},
                        {'id': 'b', 'status': 'failed', 'bbox_2d': None}]}
    assert discards_base(MISSING, _fit([0, 0, 100, 100]), repair)


def test_a_repair_asked_to_remove_an_extra_part_may_shrink() -> None:
    from shape_integration.fitting.repair_guard import discards_base
    request = [{'kind': 'extra', 'evidence': 'duplicate', 'action': 'remove'}]
    assert not discards_base(request, _fit([0, 0, 100, 100]), _fit([60, 0, 100, 40]))


def test_there_is_nothing_to_discard_without_an_old_fit() -> None:
    from shape_integration.fitting.repair_guard import discards_base
    assert not discards_base(MISSING, {'parts': []}, _fit([60, 0, 100, 40]))


class RepairKeepsOldGeometry(fixtures.unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp

    def _run(self, repaired: Dict[str, Any]) -> Dict[str, Any]:
        source = self.work / 'localization'; case = source / 'cases/case'
        save(source / 'manifest.json', {'images': [self.item]})
        save(case / 'geometry.json', {'image_size': self.item['source_size'],
                                      'coordinate_system': 'image_pixel_centers', 'objects': [self.candidate]})
        save(case / 'summary.json', {'image_coverage_status': 'complete', 'errors': {}, 'local_failures': 0})
        save(case / 'dedup/image_boxes_original.json', {'assets': []})
        provenance = fixtures.source_manifest(source, [self.item]); self.options['coverage_rounds'] = 1
        state = {'covered': False}

        def responder(tag: str, prompt: Dict[str, Any], count: int) -> Dict[str, Any]:
            if tag == 'observe':
                return repaired if 'coverage_requested_corrections' in prompt else fixtures.observed('G1')
            if tag == 'check':
                answer = fixtures.reviewed(prompt, 'accept' if state['covered'] else 'revise')
                targets = prompt.get('required_source_observations', {}).get('targets', [])
                if targets and state['covered']:
                    answer['source_observations'] = [{'id': row['id'], 'source_relation': 'supported',
                                                      'evidence': 'The requested frame is traced by the fitted part.'} for row in targets]
                return answer
            if tag == 'coverage':
                state['covered'] = True
                return {'status': 'uncertain', 'new_objects': [], 'possible_duplicates': [],
                        'uncertain': ['Explicit source-visible missing frame'],
                        'repair_existing': [{'object_id': 'G1', 'issues': [{'kind': 'missing', 'part_ids': [],
                            'evidence': 'The source has a visible frame absent from the fit', 'action': 'Recover that frame'}]}]}
            raise AssertionError(tag)
        runtime = fixtures.FakeRuntime(self.config, responder)
        with ThreadPoolExecutor(max_workers=2) as pool, patch('shape_integration.batch.case', side_effect=AssertionError):
            fixtures.run_case(self.work, self.work / 'new', self.item, runtime, self.config, pool, source, provenance, False)
        return read(self.work / 'new/cases/case/fitting/objects/G1/record.json')

    def test_a_repair_that_returns_only_the_requested_piece_is_set_aside(self) -> None:
        piece = fixtures.observed('G1'); piece['parts'] = piece['parts'][1:]
        record = self._run(piece)
        assert record['selected_revision'].endswith('/revision_baseline')
        assert record['selection_reason'] == 'repair_discarded_previous_geometry'

    def test_a_repair_that_returns_the_whole_object_replaces_the_old_fit(self) -> None:
        record = self._run(fixtures.observed('G1'))
        assert record['selected_revision'].endswith('/revision_00')
        assert record['state'] == 'accepted_by_model'
