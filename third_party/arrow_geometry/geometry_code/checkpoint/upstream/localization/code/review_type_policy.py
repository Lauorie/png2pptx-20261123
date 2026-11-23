"""Preserve an existing region's chosen rendering representation during review.

This is provenance-based inheritance, not a visual classifier. Coordinates and
labels are never inferred or rewritten. Raw model answers remain separate.
"""
import copy

VERSION = "inherit-existing-render-type-v1"


def inherit(answer, candidates):
    result = copy.deepcopy(answer)
    audit = {"policy": VERSION, "inherited": [], "blocked_mixed_type_patches": [],
             "new_region_ids": [], "coordinates_reinterpreted": False}
    if result.get("decision") == "expand":
        return result, audit
    index = {r["id"]: r for r in candidates}
    for name in ("keep_ids", "replacements", "discard", "inspect"):
        if not isinstance(result.get(name), list):
            raise ValueError(name + " array required")
    assigned = list(result["keep_ids"])
    for patch in result["replacements"]:
        if not isinstance(patch, dict) or not isinstance(patch.get("source_ids"), list):
            raise ValueError("replacement source_ids array required")
        if not isinstance(patch.get("regions"), list) or not patch["regions"]:
            raise ValueError("replacement regions must be nonempty")
        if not isinstance(patch.get("reason"), str) or not patch["reason"].strip():
            raise ValueError("Replacement needs one short reason")
        assigned.extend(patch["source_ids"])
    for discard in result["discard"]:
        if not isinstance(discard, dict) or not isinstance(discard.get("ids"), list):
            raise ValueError("discard ids required")
        assigned.extend(discard["ids"])
    if (any(not isinstance(k, str) for k in assigned) or
            len(assigned) != len(set(assigned)) or set(assigned) != set(index)):
        raise ValueError("Every input candidate needs exactly one disposition before type inheritance")

    replacements = []
    for patch in result["replacements"]:
        refs = patch["source_ids"]
        kinds = {index[k]["render_type"] for k in refs}
        if len(kinds) > 1:
            # No valid child-to-parent type mapping is supplied by this protocol.
            # Preserve the input candidates; do not choose a majority type or
            # request a new classification call.
            result["keep_ids"].extend(refs)
            audit["blocked_mixed_type_patches"].append({
                "source_ids": list(refs), "source_types": {k: index[k]["render_type"] for k in refs},
                "proposed_patch": copy.deepcopy(patch), "action": "retain_original_candidates"})
            continue
        if kinds:
            kind = next(iter(kinds))
            for region in patch["regions"]:
                if not isinstance(region, dict):
                    raise ValueError("region object required")
                audit["inherited"].append({"region_id": region.get("id"), "source_ids": list(refs),
                                           "model_render_type": region.get("render_type"),
                                           "inherited_render_type": kind})
                region["render_type"] = kind
        else:
            # New candidates retain the pre-existing classification contract.
            # The regular region validator still requires an explicit type.
            audit["new_region_ids"].extend(r.get("id") for r in patch["regions"] if isinstance(r, dict))
        replacements.append(patch)
    result["replacements"] = replacements
    return result, audit
