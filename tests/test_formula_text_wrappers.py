"""A text group inside a formula must come out as its text whatever text command wraps it."""
import pytest
from latex2mathml.converter import convert
from lxml import etree

from full_recovery.formula_text import prepare, restore


def _visible(latex: str) -> str:
    prepared, replacements = prepare(latex)
    tree = restore(etree.fromstring(convert(prepared).encode('utf-8')), replacements)
    return ''.join(tree.itertext())


def test_a_sans_serif_wrapper_around_bold_text_shows_the_text() -> None:
    visible = _visible(r'\textsf{\textbf{Robot state }}\left[\mathbf{x}^{\prime};\,\mathbf{y}^{\prime}\right]')
    assert visible.startswith('Robot state [')
    assert 'A2FTextSlot' not in visible and '\\' not in visible


@pytest.mark.parametrize('latex, shown', [
    (r'\textsc{\textit{Encoder}}+x', 'Encoder+x'),
    (r'\textup{loss}_{\textsf{total}}', 'losstotal'),
    (r'\mbox{\textbf{Step }}t', 'Step t'),
    (r'\textsf{plain words}+y', 'plain words+y'),
    (r'\texttt{a \textsf{b} c}', 'a b c'),
])
def test_no_text_wrapper_leaves_a_placeholder_or_tex_behind(latex: str, shown: str) -> None:
    assert _visible(latex) == shown


def test_the_wrapper_and_its_inner_group_become_one_placeholder_that_keeps_the_inner_style() -> None:
    prepared, replacements = prepare(r'\textsf{\textbf{Robot state }}[x]')
    assert prepared == r'\text{A2FTextSlot0}[x]'
    assert [(span[0], span[1]) for span in replacements['A2FTextSlot0']] == [('Robot state ', True)]


def test_a_command_that_is_not_a_text_group_is_still_refused_inside_text() -> None:
    with pytest.raises(ValueError):
        prepare(r'\texttt{\unknown{x}}')


def test_math_commands_outside_text_are_left_to_the_math_parser() -> None:
    prepared, replacements = prepare(r'\frac{a}{b}+\mathbf{x}')
    assert prepared == r'\frac{a}{b}+\mathbf{x}' and replacements == {}
