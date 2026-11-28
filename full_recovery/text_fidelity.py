"""How much of a figure's text the slide reproduces, as read by an independent reader.

The pixel score in ``fidelity.py`` cannot read: a label recognised as other
words, set twice on top of itself, or left out of a busy area costs it almost
nothing. Here a vision model (``text_reader.py``)
transcribes the source figure and the rendered slide, and the two
transcriptions are compared.

The comparison does not use positions, because a vision model reports them too
loosely to match text by place: whole lines are matched first, then the words
of the remaining lines, then nearly equal words are paired. Spacing and
look-alike letters, where a reader wavers, are ignored. Recall is the share of
the source's characters the slide has, precision the share of the slide's
characters the source has. Text set in the wrong place therefore still counts
as present; ``fidelity.py`` is the one that sees where things are.
"""
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

PAIRING = 0.5          # of the longer word: two left-over words differing by at most this much are one word misread
LOOK_ALIKE = str.maketrans({'I': 'l', '1': 'l', '|': 'l', 'O': '0'})


@dataclass(frozen=True)
class Difference:
    """A word the two readings disagree on; an empty string marks a word only one of them has."""

    source: str
    slide: str


@dataclass(frozen=True)
class TextFidelity:
    """Agreement between the text of a source figure and of its slide."""

    recall: float
    precision: float
    overall: float
    source_characters: int
    slide_characters: int
    differences: Tuple[Difference, ...]


def normalise(text: str) -> str:
    """Drop spacing and fold the letters a reader cannot tell apart."""
    return ''.join(text.split()).translate(LOOK_ALIKE)


def distance(first: str, second: str) -> int:
    """Return the number of single-character edits that turn ``first`` into ``second``."""
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _unmatched(first: Sequence[str], second: Sequence[str]) -> Tuple[List[str], List[str], int]:
    """Cancel items equal after normalising; return what is left of each side and the characters cancelled."""
    common = Counter(normalise(item) for item in first) & Counter(normalise(item) for item in second)
    matched = sum(len(key) * count for key, count in common.items())
    left: List[List[str]] = []
    for items in (first, second):
        budget = Counter(common)
        rest: List[str] = []
        for item in items:
            key = normalise(item)
            if budget[key] > 0:
                budget[key] -= 1
            elif key:
                rest.append(item)
        left.append(rest)
    return left[0], left[1], matched


def _paired(first: Sequence[str], second: Sequence[str]) -> Tuple[List[Difference], int]:
    """Pair nearly equal left-over words, closest first; return every difference and the characters agreeing."""
    a, b = [normalise(word) for word in first], [normalise(word) for word in second]
    candidates = sorted((distance(x, y) / max(len(x), len(y)), i, j)
                        for i, x in enumerate(a) for j, y in enumerate(b))
    used_first, used_second, agreeing = set(), set(), 0
    differences: List[Difference] = []
    for share, i, j in candidates:
        if share > PAIRING:
            break
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        agreeing += round((1 - share) * max(len(a[i]), len(b[j])))
        differences.append(Difference(first[i], second[j]))
    differences += [Difference(word, '') for i, word in enumerate(first) if i not in used_first]
    differences += [Difference('', word) for j, word in enumerate(second) if j not in used_second]
    return differences, agreeing


def compare(source: Sequence[str], slide: Sequence[str]) -> TextFidelity:
    """Compare two transcriptions, each a list of text lines in any order.

    Args:
        source: The lines read on the source figure.
        slide: The lines read on the rendered slide.

    Returns:
        Recall, precision, their harmonic mean and the words that differ.
    """
    source_total = sum(len(normalise(line)) for line in source)
    slide_total = sum(len(normalise(line)) for line in slide)
    lines_first, lines_second, by_line = _unmatched(source, slide)
    words_first, words_second, by_word = _unmatched([word for line in lines_first for word in line.split()],
                                                    [word for line in lines_second for word in line.split()])
    differences, by_pair = _paired(words_first, words_second)
    matched = by_line + by_word + by_pair
    recall = matched / source_total if source_total else 1.
    precision = matched / slide_total if slide_total else 1.
    overall = 2 * recall * precision / (recall + precision) if recall + precision else 0.
    return TextFidelity(recall, precision, overall, source_total, slide_total, tuple(differences))


def score_text(source: Image.Image, rendered: Image.Image,
               read: Callable[[Image.Image], List[str]]) -> TextFidelity:
    """Read ``source`` and ``rendered`` (brought to the source's size) and compare the readings."""
    slide = rendered if rendered.size == source.size else rendered.resize(source.size, Image.LANCZOS)
    return compare(read(source), read(slide))
