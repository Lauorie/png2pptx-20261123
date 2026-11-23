"""Cross-frame hint contract: numerical locations belong to structured fields.

This is a text-protocol guard, not a geometric or identity heuristic. It never
maps guessed prose coordinates, edits numeric arrays, or changes target IDs.
Raw hints and every withheld span remain available in a separate audit.
"""
import copy
import re

VERSION = "semantic-hints-no-unframed-positions-v1"
HINT_KEYS = frozenset({
    "label", "description", "locator", "locator_hint", "identity_hint",
    "identity_locator", "target_description", "start_hint", "end_hint",
    "route_hint", "residual_hint",
})
NUMBER = r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
APPROX = r"(?:约为?|大约|大概|≈|~|about|around|approximately)?\s*"
VALUE = APPROX + NUMBER
# Coordinate pairs / boxes, including approximation inside or outside brackets.
TUPLE = re.compile(
    APPROX + r"[\(\[（【]\s*[/,，]?\s*" + VALUE +
    r"(?:\s*[,，/]\s*" + VALUE + r"){1,}\s*[\)\]）】]", re.I)
# Axis assignments and lists/ranges; x=1, y≈2/3/4 and Chinese axis names.
AXIS = re.compile(
    r"(?<![A-Za-z0-9_])(?:x|y|横坐标|纵坐标)\s*[=≈~:：]\s*" + VALUE +
    r"(?:\s*(?:[-–—~至到/,，])\s*" + VALUE + r")*", re.I)
POSITION_PAIR = re.compile(
    r"(?:坐标|位置|像素位置|coordinates?|position|point)\s*(?:为|在|是|at|is|[:：=])?\s*" +
    VALUE + r"\s*[,，]\s*" + VALUE, re.I)
PIXEL_MEASUREMENT = re.compile(
    VALUE + r"(?:\s*[-–—~至到]\s*" + VALUE + r")?\s*(?:pixels?|px|像素)", re.I)
PATTERNS = (("coordinate_tuple", TUPLE), ("axis_assignment", AXIS),
            ("position_pair", POSITION_PAIR), ("pixel_measurement", PIXEL_MEASUREMENT))


def clean_hint(value):
    if not isinstance(value, str):
        return copy.deepcopy(value), []
    text = value
    spans = []
    for kind, pattern in PATTERNS:
        def remove(match):
            spans.append({"kind": kind, "text": match.group(0)})
            return ""
        text = pattern.sub(remove, text)
    if not spans:
        return value, []
    text = re.sub(r"[（(\[]\s*[,，;；、\s]*[)）\]]", "", text)
    text = re.sub(r"[ \t]+", " ", text).strip(" ,，;；")
    return text, spans


def clean_prompt(prompt):
    """Return a copy and an audit; normalize only named semantic string fields."""
    changes = []
    def visit(value, path):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key in HINT_KEYS and isinstance(child, str):
                    cleaned, spans = clean_hint(child)
                    result[key] = cleaned
                    if spans:
                        changes.append({
                            "path": path + [key], "before": child, "after": cleaned,
                            "withheld": spans, "reason": "unframed_numeric_positions",
                        })
                else:
                    result[key] = visit(child, path + [key])
            return result
        if isinstance(value, list):
            return [visit(child, path + [index]) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return visit(prompt, []), changes

