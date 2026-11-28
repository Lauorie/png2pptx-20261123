"""Type weight is judged against the figure's other text, which a magnified crop cannot show."""
import copy
from types import SimpleNamespace

from full_recovery.text_weight import install, settle_weight


def _reading(*flags: bool) -> dict:
    return {'id': 'T1', 'status': 'located', 'text': 'ab', 'bold': all(flags),
            'runs': [{'text': 'a', 'bold': flag} for flag in flags]}


def test_crop_reading_cannot_make_a_plain_label_bold() -> None:
    settled = settle_weight({'bold': False}, _reading(True, True))
    assert settled['bold'] is False
    assert [run['bold'] for run in settled['runs']] == [False, False]


def test_crop_reading_cannot_make_a_bold_label_plain() -> None:
    settled = settle_weight({'bold': True}, _reading(False))
    assert settled['bold'] is True
    assert [run['bold'] for run in settled['runs']] == [True]


def test_weights_told_apart_inside_one_label_are_kept() -> None:
    second = _reading(True, False)
    assert settle_weight({'bold': False}, second) == second


def test_agreeing_readings_are_untouched() -> None:
    second = _reading(True)
    assert settle_weight({'bold': True}, second) == second


def test_without_a_whole_figure_reading_the_crop_reading_stands() -> None:
    second = _reading(True)
    assert settle_weight({}, second) == second


def test_a_label_without_runs_is_settled_on_its_own_flag() -> None:
    assert settle_weight({'bold': False}, {'id': 'T1', 'bold': True})['bold'] is False


def test_readings_are_not_modified_in_place() -> None:
    first, second = {'bold': False}, _reading(True, True)
    before = copy.deepcopy(second)
    settle_weight(first, second)
    assert second == before


def test_installed_validator_settles_each_located_label() -> None:
    def validate_attributes(raw, sources, locked, size, *, partial=False):
        return {'items': copy.deepcopy(raw['items'])}
    module = SimpleNamespace(validate_attributes=validate_attributes)
    install(module)
    install(module)  # installing twice changes nothing more
    raw = {'items': [_reading(True), {'id': 'T2', 'status': 'ambiguous'}]}
    result = module.validate_attributes(raw, [{'id': 'T1', 'bold': False}, {'id': 'T2', 'bold': True}], {}, (10, 10))
    assert result['items'][0]['bold'] is False
    assert result['items'][1] == {'id': 'T2', 'status': 'ambiguous'}
