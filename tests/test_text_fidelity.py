"""Text fidelity compares what an independent reader reads on the source and on the slide."""
from full_recovery.text_fidelity import compare, distance, normalise


def _pairs(result):
    return sorted((d.source, d.slide) for d in result.differences)


def test_spacing_and_look_alike_letters_do_not_count_as_differences() -> None:
    assert normalise('2D lmage') == normalise('2DImage')
    assert normalise('(Fig. 4)') == normalise('(Fig.4)')
    assert normalise('O0') == normalise('0O')


def test_letter_case_and_real_letters_still_count() -> None:
    assert normalise('Loss') != normalise('loss')
    assert normalise('Val.Error') != normalise('Valve:Junior')


def test_distance_counts_single_character_edits() -> None:
    assert distance('kitten', 'sitting') == 3
    assert distance('', 'abc') == 3 and distance('abc', 'abc') == 0


def test_identical_text_scores_one_whatever_the_order() -> None:
    lines = ['Semantic Encoder', 'Regression Head']
    result = compare(lines, lines[::-1])
    assert result.recall == result.precision == result.overall == 1.0 and result.differences == ()


def test_a_line_spaced_differently_is_the_same_text() -> None:
    assert compare(['Val. Error', 'Classification Branch'], ['Val.Error', 'ClassificationBranch']).overall == 1.0


def test_a_misread_label_lowers_the_score_and_is_listed() -> None:
    result = compare(['Val.Error', 'Skill Space'], ['Valve: Junior', 'Skill Space'])
    assert result.overall < 0.9
    assert _pairs(result) == [('', 'Junior'), ('', 'Valve:'), ('Val.Error', '')]


def test_text_missing_from_the_slide_costs_recall_only() -> None:
    result = compare(['Semantic Encoder', 'Regression Head'], ['Semantic Encoder'])
    assert result.recall < 0.6 and result.precision == 1.0
    assert _pairs(result) == [('Head', ''), ('Regression', '')]


def test_text_only_the_slide_has_costs_precision_only() -> None:
    result = compare(['Semantic Encoder'], ['Semantic Encoder', 'ghost'])
    assert result.recall == 1.0 and result.precision < 0.8
    assert _pairs(result) == [('', 'ghost')]


def test_one_line_read_as_two_pieces_is_still_the_same_text() -> None:
    whole, pieces = ['instance-centric dense object detection'], ['instance-centric dense', 'object detection']
    assert compare(whole, pieces).overall == 1.0 and compare(pieces, whole).overall == 1.0


def test_a_label_the_slide_shows_twice_is_extra_text() -> None:
    result = compare(['Query', 'Key'], ['Query', 'Query', 'Key'])
    assert result.recall == 1.0 and result.precision < 1.0 and _pairs(result) == [('', 'Query')]


def test_a_slightly_wrong_word_is_paired_with_the_right_one_and_keeps_part_of_its_credit() -> None:
    result = compare(['gradient direction'], ['gradient direktion'])
    assert _pairs(result) == [('direction', 'direktion')]
    assert 0.9 < result.overall < 1.0


def test_unrelated_words_are_not_paired() -> None:
    assert _pairs(compare(['alpha'], ['query'])) == [('', 'query'), ('alpha', '')]


def test_a_figure_without_text_scores_one() -> None:
    result = compare([], [])
    assert result.overall == 1.0 and result.source_characters == 0
