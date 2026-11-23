"""Stage-local adapters: output syntax is separate from visual decisions.

User-authorized boundary clipping preserves raw model coordinates separately.
Invalid observations stay in the audit. Review replacements are atomic per source group.
"""
import copy
import json
import math
import re
from collections import Counter
import contracts as C

VERSION = "sam-candidate-protocol-v3-clamp"
META = {"task", "notes", "comment", "explanation"}
FORBIDDEN = {"existing_ids", "source_ids", "replace", "replacements", "discard",
             "delete", "updates", "edits", "keep_ids"}
REGION_FIELDS = {"bbox", "render_type", "grouping", "label", "extent", "id"}


def bound_region(raw, size):
    """Clip to this input's pixel centers; never reorder or invent a box."""
    if not isinstance(raw, dict):
        raise ValueError("Candidate must be an object")
    box = raw.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox needs four LOCAL pixel coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError("bbox needs finite JSON numbers, not bool/string")
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError("bbox corners must be ordered with positive area")
    limits = [size[0]-1, size[1]-1, size[0]-1, size[1]-1]
    bounded = [min(max(v, 0), upper) for v, upper in zip(box, limits)]
    if not (bounded[0] < bounded[2] and bounded[1] < bounded[3]):
        raise ValueError("bbox has no positive-area intersection with the input image")
    row = copy.deepcopy(raw)
    row.pop("raw_model_bbox", None)
    row["bbox"] = bounded
    adjustment = None
    if bounded != box:
        row["raw_model_bbox"] = copy.deepcopy(box)
        adjustment = dict(raw_bbox=copy.deepcopy(box), bounded_bbox=bounded,
                          lower_xy=[0, 0], upper_xy=[size[0]-1, size[1]-1],
                          reason="clip_to_input_pixel_centers")
    return row, adjustment


class StageProtocol:
    visual_format_retries = 0

    def __init__(self, size, stage):
        self.size = tuple(size)
        self.stage = stage

    def parse(self, content, repair_log=None):
        from transport.model import close_object_before_array
        from transport.common import write_json
        text = content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # The fixed prompt may elicit a named JSON array rather than an object.
        # Accept a unique explicit assignment, including nonempty arrays; never
        # infer an empty result from absence of a field or from ordinary prose.
        assignment = re.match(r'^regions\s*=\s*', text) if self.stage == 'missing' else None
        if assignment:
            try:
                value, end = json.JSONDecoder().raw_decode(text, assignment.end())
            except json.JSONDecodeError:
                raise ValueError('Incomplete regions assignment') from None
            tail = text[end:].strip()
            if (not isinstance(value, list) or re.search(r'[{}\[\]]', tail) or
                re.search(r'\bregions\s*=', tail)):
                raise ValueError('Ambiguous regions assignment')
            if repair_log:
                write_json(repair_log, {'repair':'wrap_explicit_named_array',
                                       'stage':self.stage,'coordinates_changed':False})
            return {'regions': value}
        keys = {"regions"} if self.stage == "missing" else {"keep_ids", "replacements", "discard"}
        decoder = json.JSONDecoder()
        found = []
        cursor = 0
        while cursor < len(text):
            start = text.find("{", cursor)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                # Never accept a nested fragment of an incomplete outer answer.
                break
            cursor = end
            if (isinstance(obj, dict) and
                not {"output_schema", "response_format", "instructions", "input_size"}.intersection(obj) and
                (keys.intersection(obj) or obj.get("decision") in ("expand", "uncertain"))):
                found.append(obj)
        if len(found) == 1:
            if repair_log:
                write_json(repair_log, {"repair": "unique_stage_answer_from_wrapper",
                                        "stage": self.stage, "coordinates_changed": False})
            return found[0]
        if len(found) > 1:
            raise ValueError("Multiple possible stage answers; do not guess")
        repaired, inserted = close_object_before_array(text)
        if inserted:
            result = json.loads(repaired)
            if repair_log:
                write_json(repair_log, {"repair": "missing_object_closer",
                                        "insertions": inserted, "coordinates_changed": False})
            return result
        raise ValueError("No unambiguous complete JSON answer for " + self.stage)

    def header(self, answer, allowed):
        if not isinstance(answer, dict):
            raise ValueError("Answer must be an object")
        # Idempotent validation of a saved result; raw model outputs are retained
        # separately and adapters do not trust caller-supplied acceptance audits.
        if answer.get("frame", "LOCAL") != "LOCAL":
            raise ValueError("Explicit non-LOCAL frame cannot be reinterpreted")
        unknown = set(answer) - allowed - META
        if unknown:
            raise ValueError("Unsupported stage fields: " + str(sorted(unknown)))
        return dict(version=VERSION, stage=self.stage, coordinate_frame="LOCAL",
                    filled_frame="frame" not in answer,
                    ignored_metadata={k: copy.deepcopy(answer[k]) for k in META if k in answer},
                    accepted=[], rejected=[], raw_coordinates_changed=False,
                    effective_coordinates_changed=False, bbox_adjustments=[],
                    bbox_policy="clip_to_input_pixel_centers",
                    raw_answer=copy.deepcopy(answer))

    def validate_saved(self, answer):
        return self(copy.deepcopy(answer['_protocol']['raw_answer']))

    def repair_audit(self, original, repaired):
        if repaired.get("format_repair_failed"):
            raise ValueError("Text repair did not recover a definite answer")
        # Fixed metadata may change; actual object payloads, coordinates, visual
        # decisions and per-object associations must remain exactly equal.
        before = self.parse(original)
        def semantic(obj):
            obj = copy.deepcopy(obj)
            if obj.get("frame", "LOCAL") != "LOCAL":
                raise ValueError("Cannot repair an explicitly different frame")
            for k in META | {"frame", "_protocol"}:
                obj.pop(k, None)
            return obj
        if semantic(before) != semantic(repaired):
            raise ValueError("Text repair changed visual payload or decisions")
        return {"visual_payload_preserved": True, "coordinates_changed": False,
                "fixed_metadata_only": True, "images_sent": 0}


class MissingProtocol(StageProtocol):
    def __init__(self, size):
        super().__init__(size, "missing")

    def __call__(self, answer):
        audit = self.header(answer, {"frame", "decision", "reason", "regions", "uncertain"})
        decision = answer.get("decision", "add")
        if decision not in ("add", "expand"):
            raise ValueError("Missing stage cannot perform edits")
        if not isinstance(answer.get("regions"), list):
            raise ValueError("Explicit regions array required; missing is not empty")
        if decision == "expand":
            if answer["regions"] or not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
                raise ValueError("Expansion requires a reason and no submitted regions")
            return dict(frame="LOCAL", decision="expand", reason=answer["reason"],
                        regions=[], uncertain=[], _protocol=audit)
        rows = []
        for i, raw in enumerate(answer["regions"]):
            try:
                if not isinstance(raw, dict):
                    raise ValueError("Candidate must be an object")
                if set(raw) - REGION_FIELDS:
                    raise ValueError("Unsupported candidate fields; no old-object edits")
                # All discovery outputs are observations awaiting review, including
                # historical complete/clipped labels. No false complete promotion.
                row = {k: copy.deepcopy(raw[k]) for k in ("bbox", "render_type", "grouping", "label")}
                row, adjustment = bound_region(row, self.size)
                row["extent"] = "uncertain"
                C.regions_valid([dict(row, id="new_" + str(i))], self.size)
                row.update(review_confirmed=False, candidate_status="unreviewed")
                if "extent" in raw:
                    row["model_extent_hint"] = copy.deepcopy(raw["extent"])
                rows.append(row)
                if adjustment:
                    audit["bbox_adjustments"].append(dict(adjustment, source_index=i))
                    audit["effective_coordinates_changed"] = True
                audit["accepted"].append(dict(source_index=i, raw_bbox=copy.deepcopy(raw["bbox"]),
                                                accepted_bbox=copy.deepcopy(row["bbox"])))
            except (ValueError, TypeError, KeyError) as exc:
                audit["rejected"].append(dict(source_index=i, raw=copy.deepcopy(raw), reason=str(exc)))
        notes = answer.get("uncertain", [])
        if not isinstance(notes, list) or any(not isinstance(s, str) for s in notes):
            raise ValueError("uncertain, when present, must be a text array")
        return dict(frame="LOCAL", decision="add", reason=answer.get("reason", ""),
                    regions=rows, uncertain=copy.deepcopy(notes), _protocol=audit)


class ReviewProtocol(StageProtocol):
    def __init__(self, size, focus):
        super().__init__(size, "review")
        self.focus = {r["id"]: r for r in focus}

    def __call__(self, answer):
        audit = self.header(answer, {"frame", "decision", "reason", "keep_ids", "replacements", "discard"})
        audit["raw_answer"] = copy.deepcopy(answer)
        decision = answer.get("decision")
        if decision in ("expand", "uncertain"):
            if not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
                raise ValueError("A context or uncertainty reason is required")
            return dict(frame="LOCAL", decision=decision, reason=answer["reason"], _protocol=audit)
        if decision != "edit":
            raise ValueError("Review requires edit, expand, or uncertain")
        for k in ("keep_ids", "replacements", "discard"):
            if not isinstance(answer.get(k), list):
                raise ValueError(k + " array required")
        operations = []
        for ident in answer["keep_ids"]:
            operations.append(("keep", [ident], ident))
        for patch in answer["replacements"]:
            operations.append(("replace", patch.get("source_ids") if isinstance(patch, dict) else None, patch))
        for item in answer["discard"]:
            operations.append(("discard", item.get("ids") if isinstance(item, dict) else None, item))
        counts = Counter(ident for _, refs, _ in operations if isinstance(refs, list)
                         for ident in refs if isinstance(ident, str))
        conflicting = {k for k, n in counts.items() if n > 1}
        accepted_ids = set()
        out = dict(frame="LOCAL", decision="edit", reason=answer.get("reason", ""),
                   keep_ids=[], replacements=[], discard=[])
        for i, (kind, refs, raw) in enumerate(operations):
            try:
                if (not isinstance(refs, list) or not refs or
                    any(not isinstance(x, str) for x in refs)):
                    raise ValueError("Operation must reference explicit candidate IDs")
                if not set(refs) <= set(self.focus):
                    raise ValueError("Unknown or read-only IDs cannot be edited")
                if conflicting.intersection(refs):
                    raise ValueError("Conflicting dispositions for candidate IDs")
                if kind == "replace":
                    if set(raw) - {"source_ids", "regions", "reason"}:
                        raise ValueError("Unsupported replacement fields")
                    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
                        raise ValueError("Replacement needs visible evidence")
                    if not raw.get("regions"):
                        raise ValueError("Empty replacement cannot silently delete a candidate")
                    patch = copy.deepcopy(raw)
                    patch["regions"] = []
                    adjustments = []
                    for j, region in enumerate(raw["regions"]):
                        bounded, adjustment = bound_region(region, self.size)
                        patch["regions"].append(bounded)
                        if adjustment:
                            adjustments.append(dict(adjustment, operation=i, region_index=j,
                                                    source_ids=copy.deepcopy(refs)))
                    C.regions_valid(patch["regions"], self.size)
                    # A source group is atomic: never consume it with only some
                    # children accepted, which could lose a split branch.
                    out["replacements"].append(patch)
                    audit["bbox_adjustments"].extend(adjustments)
                    audit["effective_coordinates_changed"] |= bool(adjustments)
                elif kind == "discard":
                    if set(raw) - {"ids", "reason"}:
                        raise ValueError("Unsupported discard fields")
                    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
                        raise ValueError("Discard requires a visible reason")
                    out["discard"].append(copy.deepcopy(raw))
                else:
                    out["keep_ids"].append(raw)
                accepted_ids.update(refs)
                audit["accepted"].append(dict(operation=i, kind=kind, source_ids=refs))
            except (ValueError, TypeError, KeyError) as exc:
                audit["rejected"].append(dict(operation=i, kind=kind, raw=copy.deepcopy(raw),
                                               source_ids=copy.deepcopy(refs), reason=str(exc)))
        pending = sorted(set(self.focus) - accepted_ids)
        audit["pending_focus_ids"] = pending
        out["_protocol"] = audit
        return out

