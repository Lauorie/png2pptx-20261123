"""Tolerate one observed text-inventory defect: a run that lost its "text" key.

Some models write a run whose text starts with punctuation as
``{", y", "script": "normal", ...}`` instead of
``{"text": ", y", "script": "normal", ...}``.  The value is intact and only the
key is missing, so the key can be restored without guessing any content.

The repair is driven by the JSON decoder's own error position, so it never
touches text that already parses and never rewrites anything inside a string.
Every other defect is left alone and still fails as before.
"""
import json
import logging
import re
from typing import Any, Tuple

logger = logging.getLogger(__name__)

PROTOCOL = 'run_text_key_restoration.1'
MAX_RESTORED_KEYS = 500
_KEYLESS_STRING = re.compile(r'\{\s*("(?:[^"\\]|\\.)*")\s*')
_SCRIPT_MEMBER = re.compile(r'\s*,\s*"script"\s*:')


def _keyless_string_start(text: str, colon_at: int) -> int:
    """Return where the keyless string starts, or -1 if this is another defect.

    Args:
        text: The reply being decoded.
        colon_at: Position where the decoder expected ``:`` after an object key.

    Returns:
        Index of the opening quote of the string that should be a ``text`` value.
    """
    if not _SCRIPT_MEMBER.match(text, colon_at):
        return -1
    brace = text.rfind('{', 0, colon_at)
    while brace >= 0:
        found = _KEYLESS_STRING.fullmatch(text, brace, colon_at)
        if found:
            return found.start(1)
        # The string itself may contain a brace; try the one before it.
        brace = text.rfind('{', 0, brace)
    return -1


def restore_run_text_keys(text: str) -> Tuple[str, int]:
    """Restore missing ``"text"`` keys of runs in an otherwise valid reply.

    Args:
        text: JSON text returned by the model.

    Returns:
        The (possibly repaired) text and the number of keys restored.  Text
        that already parses is returned unchanged with a count of zero.
    """
    restored = 0
    while restored < MAX_RESTORED_KEYS:
        try:
            json.loads(text)
            break
        except json.JSONDecodeError as exc:
            if not exc.msg.startswith("Expecting ':' delimiter"):
                break
            start = _keyless_string_start(text, exc.pos)
            if start < 0:
                break
            text = text[:start] + '"text": ' + text[start:]
            restored += 1
    return text, restored


def loads(text: str) -> Any:
    """Decode JSON, restoring dropped run ``text`` keys when that is the defect.

    Args:
        text: JSON text returned by the model.

    Returns:
        The decoded value.

    Raises:
        json.JSONDecodeError: When the text is malformed in any other way.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired, restored = restore_run_text_keys(text)
        if not restored:
            raise
        value = json.loads(repaired)
        logger.warning('Restored %d dropped run "text" key(s) in a text reply (%s)',
                       restored, PROTOCOL)
        return value


class _TolerantJson:
    """The ``json`` module with ``loads`` replaced by the tolerant decoder."""

    loads = staticmethod(loads)

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)


def install() -> None:
    """Make the text stage's live and cached reply readers use ``loads``."""
    from shape_integration import transport
    from shape_integration.texts import inventory_recovery, staged_formula
    if getattr(transport, '_run_text_key_protocol', None) == PROTOCOL:
        return

    def parse_json(text: str) -> Any:
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        return loads(text)

    transport.parse_json = parse_json
    inventory_recovery.json = _TolerantJson()
    staged_formula.json = _TolerantJson()
    transport._run_text_key_protocol = PROTOCOL
