"""Limit model-facing pixel coordinates; never quantize pipeline geometry in place."""
import copy
import json
import math
import re

VERSION = "model-input-coordinates-1dp-v1"
_COORDINATES = {
    "xy", "x", "y", "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax",
    "points", "point", "nodes", "start", "end", "via", "vertices", "contour",
    "contours", "polygon", "polyline", "control_points", "position", "anchor",
    "anchors", "crop_in_context", "local_region_in_global_px",
}
# Format-only repairs must preserve the original reply, including its numbers.
_VERBATIM = {
    "original_reply", "previous_output", "raw_reply", "original_output",
    "validation_error", "error", "format_repair",
}
_BARE_BOUNDS = re.compile(
    r"(?:GLOBAL\s*观察范围|LOCAL_REGION_IN_GLOBAL_PX|bounds_global|bbox)"
    r"\s*[:：=]\s*$", re.I)


def _coordinate_key(key):
    key = key.lower()
    return key in _COORDINATES or "bbox" in key.split("_") or "bounds" in key.split("_")


def _number(value):
    if type(value) is float and math.isfinite(value):
        result = round(value, 1)
        return 0.0 if result == 0 else result
    return value


def _structured(value, key=""):
    if isinstance(value, dict):
        return {k: copy.deepcopy(v) if k in _VERBATIM else _structured(v, k)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        result = [_structured(child, key) for child in value]
        # Rounding must not turn a valid thin box into an empty box.
        if ("bbox" in key.lower().split("_") or "bounds" in key.lower().split("_")) and len(value) == 4:
            if all(type(v) in (int, float) and math.isfinite(v) for v in value):
                for lo, hi in ((0, 2), (1, 3)):
                    if value[lo] < value[hi] and result[lo] >= result[hi]:
                        result[lo] = math.floor(value[lo] * 10) / 10
                        result[hi] = math.ceil(value[hi] * 10) / 10
        return tuple(result) if isinstance(value, tuple) else result
    return _number(value) if _coordinate_key(key) else copy.deepcopy(value)


def normalize_prompt(prompt):
    """Copy a structured prompt, or normalize JSON coordinates embedded in prose.

    Do not regex-replace decimal numbers in arbitrary text: IDs, labels, scales,
    inference parameters and quoted model replies must retain their precision.
    """
    if not isinstance(prompt, str):
        return _structured(prompt)
    decoder = json.JSONDecoder()
    pieces, cursor, copied = [], 0, 0
    while cursor < len(prompt):
        match = re.search(r"[\[{]", prompt[cursor:])
        if match is None:
            break
        start = cursor + match.start()
        try:
            value, end = decoder.raw_decode(prompt, start)
        except ValueError:
            cursor = start + 1
            continue
        key = "bounds" if _BARE_BOUNDS.search(prompt[max(0, start - 80):start]) else ""
        normalized = _structured(value, key)
        if normalized != value:
            pieces.append(prompt[copied:start])
            pieces.append(json.dumps(normalized, ensure_ascii=False,
                                     indent=2 if "\n" in prompt[start:end] else None))
            copied = end
        cursor = end
    pieces.append(prompt[copied:])
    return "".join(pieces)

