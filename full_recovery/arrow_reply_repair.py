"""Tolerate two observed format slips in arrow-geometry replies.

1. Node ids written without quotes: ``[p1,507.0,830.0]``.
2. A point table without ids, ``[[583.0,300.0],...]``, whose segments refer to
   the points by position: ``"n0"``, ``"n1"``, ...

Both keep every coordinate the model returned; only the node naming is
restored.  Anything ambiguous is left unchanged and still fails validation.

This file is copied into the frozen arrow code as ``transport/reply_repair.py``
(see :func:`install`), so it must not import anything outside the standard library.
"""
import copy
import json
import re
from pathlib import Path
from typing import Any, List, Optional

MAX_QUOTED_IDS = 2000
_BARE_ID = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?=\s*,)')
_POSITIONAL = re.compile(r'([A-Za-z_]*)(\d+)')
_IMPORT_ANCHOR = 'from .prompts import SYSTEM\n'
_PARSE_ANCHOR = 'def parse(content, repair_log=None):\n'
_PARSE_WRAPPER = (
    'def parse(content, repair_log=None):\n'
    '    # Restore node naming only; coordinates are never changed (see reply_repair).\n'
    '    return reply_repair.name_positional_nodes(_parse(reply_repair.quote_node_ids(content), repair_log))\n'
    '\n\n'
    'def _parse(content, repair_log=None):\n')


def quote_node_ids(content: str) -> str:
    """Quote bare identifiers that open an array, e.g. ``[p1,507.0,830.0]``.

    Driven by the JSON decoder's own error position, so text that already
    parses, JSON literals and string contents are never touched.

    Args:
        content: Reply text.

    Returns:
        The repaired text, or the original text when this is not the defect.
    """
    text = content
    for _ in range(MAX_QUOTED_IDS):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as exc:
            found = _BARE_ID.match(text, exc.pos)
            if (not exc.msg.startswith('Expecting value') or not found
                    or found.group(1) in ('true', 'false', 'null')
                    or not text[:exc.pos].rstrip().endswith('[')):
                return content
            text = text[:found.start(1)] + '"' + found.group(1) + '"' + text[found.end(1):]
    return content


def _segment_references(answer: dict) -> Optional[List[Any]]:
    """Return every node reference used by the answer's segments, or None if malformed."""
    references = []
    for route in answer['routes']:
        if not isinstance(route, dict):
            return None
        paths = route.get('paths') if 'paths' in route else [route]
        for path in paths if isinstance(paths, list) else []:
            for segment in path.get('segments', []) if isinstance(path, dict) else []:
                if not isinstance(segment, dict) or not isinstance(segment.get('nodes'), list):
                    return None
                references.extend(segment['nodes'])
    return references


def name_positional_nodes(answer: Any) -> Any:
    """Give ids to an id-less point table that segments reference by position.

    Applies only when every point is a bare ``[x, y]`` pair, every reference is
    a name ending in a number with one common prefix, and the numbers are
    exactly ``0..k-1`` or ``1..k`` for the ``k`` points, so the numbering base
    is established by the reply itself.

    Args:
        answer: Decoded reply; not modified.

    Returns:
        A repaired copy, or ``answer`` itself when the form does not apply.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get('nodes'), list) \
            or not isinstance(answer.get('routes'), list) or not answer['nodes']:
        return answer
    nodes = answer['nodes']
    if not all(isinstance(n, list) and len(n) == 2
               and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in n) for n in nodes):
        return answer
    references = _segment_references(answer)
    if not references or not all(isinstance(r, str) for r in references):
        return answer
    matches = [_POSITIONAL.fullmatch(r) for r in references]
    if not all(matches) or len({m.group(1) for m in matches}) != 1:
        return answer
    numbers = {int(m.group(2)) for m in matches}
    count = len(nodes)
    if numbers == set(range(count)):
        base = 0
    elif numbers == set(range(1, count + 1)):
        base = 1
    else:
        return answer
    names = {int(m.group(2)): m.group(0) for m in matches}
    if len(set(names.values())) != count:
        return answer
    repaired = copy.deepcopy(answer)
    repaired['nodes'] = [[names[index + base], *xy] for index, xy in enumerate(nodes)]
    return repaired


def install(geometry_code: Path) -> None:
    """Wrap the frozen arrow parser of one prepared run with these repairs.

    Args:
        geometry_code: The run's own copy of the arrow ``geometry_code`` directory.

    Raises:
        ValueError: When the parser does not have the expected, unpatched form.
    """
    transport = Path(geometry_code)/'transport'
    model = transport/'model.py'
    source = model.read_text()
    if source.count(_IMPORT_ANCHOR) != 1 or source.count(_PARSE_ANCHOR) != 1 or 'reply_repair' in source:
        raise ValueError('Unexpected arrow parser source: ' + str(model))
    source = source.replace(_IMPORT_ANCHOR, _IMPORT_ANCHOR + 'from . import reply_repair\n')
    model.write_text(source.replace(_PARSE_ANCHOR, _PARSE_WRAPPER))
    (transport/'reply_repair.py').write_text(Path(__file__).read_text())
