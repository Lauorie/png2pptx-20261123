"""Separate reconstruction regions from discovery observations."""
from frames import native_box
from semantic_hint_contract import clean_hint, VERSION as HINT_CONTRACT


def export(case_id, regions, view, source_path, defer_uncertain=False):
    tasks = []; audits = []; deferred = []
    for i, r in enumerate(regions, 1):
        coords = native_box(r["bbox"], view)
        hint, withheld = clean_hint(r["label"])
        audits.append({"id": r["id"], **coords, "raw_hint": r["label"],
                       "semantic_hint": hint, "withheld_hint_positions": withheld})
        block = r["render_type"] == "block_arrow"
        task = {
            "target_id": f"R{i:04d}", "display_id": i,
            "region_id": r["id"], "bbox_native": coords["bbox_native"],
            "bbox_global_input_px": r["bbox"],
            "bbox_native_unclipped": coords["mapped_native_unclipped"],
            "identity_hint": hint, "kind": "single" if r["grouping"] == "single" else "dense_region",
            "grouping": r["grouping"], "render_type": r["render_type"],
            "geometry_route": "closed_shape" if block else "centerline",
            "source_image": str(source_path), "target_image": str(source_path),
            "task_role": "complete_region", "render_owner": True,
            "readonly_identities": [], "bbox_is_hint_not_pixel_clip": True,
            "requires_review": r["extent"] != "complete",
            "extent": r["extent"],
            "observation_ids": r.get("observation_ids", []),
            "instruction": "依据原图恢复此框指代的完整区域。框是位置提示，不是裁剪或禁绘掩码；保留实际可见端点和全部分支，不能按bbox重叠推断连通。",
            "block_recovery": {
                "modes": ["glm_closed_shape", "crop_sam_mask_contour"],
                "choice": "model_review", "sam_masks_are_optional_evidence": True,
                "contour_preserve_holes": True, "executed": False
            } if block else None,
        }
        if defer_uncertain and r["extent"] == "uncertain":
            task.update(task_role="missing_candidate", render_owner=False,
                        deferred_reason="localization_extent_uncertain",
                        instruction="首轮几何不恢复此候选。初次几何完成后，对照原图与实际已恢复几何逐框核验；仅确认真实遗漏才新增，仍不确定则保留待查。")
            deferred.append(task)
        else:
            tasks.append(task)
    return {"schema_version": "model-led-regions-v1", "case_id": case_id, "tasks": tasks,
            "deferred_candidates": deferred,
            "uncertain_routing": "post_geometry_missing" if defer_uncertain else "initial_geometry",
            "coordinate_frame": "native image pixel centers; exact mapping retained separately",
            "geometry_executed": False, "physical_completeness_measured": False,
            "hint_contract": HINT_CONTRACT,
            "policy": "Rendering tasks own initial geometry. Deferred candidates require a missing-only check against actual geometry; their boxes never establish coverage."}, audits
