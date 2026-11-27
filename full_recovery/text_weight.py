"""Keep the whole-figure reading of type weight when a crop reading disputes it.

Text attributes are read twice: once on the whole figure and once on a crop
that is magnified until a 14 px label is over 100 px tall. Whether lettering
is bold is a judgement against the other lettering of the figure, and the
crop no longer shows it: on 21 figures the crop reading turned 111 plain
labels bold, and their stroke thickness on the source matched plain text.
The crop reading keeps its say where it does have a reference, namely when it
tells bold and plain runs apart inside one label.
"""
import copy
import logging
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

PROTOCOL = 'whole_figure_weight.1'


def _flags(reading: Mapping[str, Any]) -> List[bool]:
    runs = reading.get('runs') or []
    flags = [bool(run['bold']) for run in runs if isinstance(run, Mapping) and 'bold' in run]
    return flags or [bool(reading.get('bold'))]


def settle_weight(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    """Settle the weight of one label read on the whole figure and again on a crop.

    Args:
        first: The whole-figure reading; only its ``bold`` is used.
        second: The crop reading, with an item-level ``bold`` and optional runs.

    Returns:
        The crop reading, with the whole-figure weight where the crop reading
        gave one uniform weight that contradicts it. Inputs are not modified.
    """
    whole: Optional[bool] = first.get('bold') if isinstance(first.get('bold'), bool) else None
    flags = _flags(second)
    if whole is None or len(set(flags)) != 1 or flags[0] == whole:
        return dict(second)
    settled = copy.deepcopy(dict(second))
    settled['bold'] = whole
    for run in settled.get('runs') or []:
        if isinstance(run, dict):
            run['bold'] = whole
    return settled


def install(two_stage_crop: Any = None) -> None:
    """Settle weight on every crop reading the text backend validates.

    Args:
        two_stage_crop: The backend's ``two_stage_crop`` module; imported when omitted.
    """
    if two_stage_crop is None:
        from shape_integration.texts import two_stage_crop
    if getattr(two_stage_crop, '_weight_protocol', None) == PROTOCOL:
        return
    original = two_stage_crop.validate_attributes

    def validate_attributes(raw, sources, *args, **kwargs):
        result = original(raw, sources, *args, **kwargs)
        first = {row['id']: row for row in sources}
        result['items'] = [
            settle_weight(first[item['id']], item)
            if item.get('status') == 'located' and item.get('id') in first else item
            for item in result['items']]
        return result

    two_stage_crop.validate_attributes = validate_attributes
    two_stage_crop._weight_protocol = PROTOCOL
