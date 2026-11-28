"""Label a replacement region from its own patch reason when the model omits one.

Candidate labels are withheld from the review prompt, and some models then leave
``label`` out of the regions they return. The patch already carries the model's
short visual reason for those regions, so that text is used as the label instead
of discarding an otherwise valid reply. Coordinates, types and written labels
are never changed, and the raw model answer stays separate.
"""
import copy
from typing import Any, Dict, List, Tuple

VERSION = "label-from-patch-reason-v1"


def _blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def fill(answer: Any) -> Tuple[Any, Dict[str, Any]]:
    """Return a copy of ``answer`` whose unlabelled replacement regions are labelled.

    Args:
        answer: One parsed review reply; malformed replies are returned untouched
            so that the regular validator reports them.

    Returns:
        The labelled copy and an audit listing the region IDs that were filled.
    """
    filled: List[Any] = []
    audit: Dict[str, Any] = {"policy": VERSION, "filled": filled}
    if not isinstance(answer, dict) or not isinstance(answer.get("replacements"), list):
        return answer, audit
    result = copy.deepcopy(answer)
    for patch in result["replacements"]:
        if not isinstance(patch, dict) or _blank(patch.get("reason")):
            continue
        if not isinstance(patch.get("regions"), list):
            continue
        for region in patch["regions"]:
            if isinstance(region, dict) and _blank(region.get("label")):
                region["label"] = patch["reason"].strip()
                filled.append(region.get("id"))
    return result, audit
