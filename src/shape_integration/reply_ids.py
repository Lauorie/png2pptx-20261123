"""Tell apart elements a reply lists under one name.

The model names what it finds. Repeated elements, such as two rows of the
same tokens, sometimes get one name each, and the protocol validator rejects
the whole reply for it, twice in a row on one figure. The elements are
distinct and correctly placed; only the names clash.
"""
import copy
from typing import Any, Callable


def distinct_ids(raw: Any, key: str) -> Any:
    """Give every listed element its own id.

    Args:
        raw: A parsed reply; anything not shaped as expected is returned unchanged.
        key: The name of the list of elements in the reply.

    Returns:
        The reply itself when its ids are distinct, otherwise a copy in which a
        later element with a used id is renamed, or dropped when it repeats an
        earlier element's id and box exactly.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get(key), list):
        return raw
    items = raw[key]
    if not all(isinstance(item, dict) and isinstance(item.get('id'), str) for item in items):
        return raw
    used = {item['id'] for item in items}
    if len(used) == len(items):
        return raw
    seen = {}
    result = []
    for item in items:
        ident = item['id']
        if ident not in seen:
            seen[ident] = [item.get('bbox_2d')]
            result.append(copy.deepcopy(item))
            continue
        if item.get('bbox_2d') in seen[ident]:
            continue
        seen[ident].append(item.get('bbox_2d'))
        number = len(seen[ident])
        while f'{ident}_{number}' in used:
            number += 1
        renamed = copy.deepcopy(item)
        renamed['id'] = f'{ident}_{number}'
        used.add(renamed['id'])
        result.append(renamed)
    return {**raw, key: result}


def with_distinct_ids(validator: Callable[[Any], Any], key: str) -> Callable[[Any], Any]:
    """Wrap a reply validator so that it checks the reply after its ids are made distinct."""
    def check(raw: Any) -> Any:
        return validator(distinct_ids(raw, key))
    return check
