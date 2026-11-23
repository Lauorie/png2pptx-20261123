#!/usr/bin/env python3
"""Model-led discovery, local coverage, region correction, and handoff."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
import contracts as C
import draft_prompts as Draft
from frames import global_frame, local_views, visible_global_bounds, crop_from_global
from presentation import now, publish, latest, compact, overlay
from handoff import export


def prepare(root, baseline, code, config, constraints, selected=None):
    if root.exists():
        raise ValueError("Run exists; use run to resume without changing frozen code")
    cfg = read(config)
    if cfg["thinking"] != {"type": "enabled"} or "reasoning_effort" in cfg or cfg["max_tokens"] != 113000:
        raise ValueError("Keep default thinking and the authorized output ceiling")
    root.mkdir(parents=True)
    shutil.copytree(code, root/"code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copytree(constraints, root/"constraints_snapshot")
    known = read(baseline/"cases.json")
    ids = selected or known
    if len(set(ids)) != len(ids) or not set(ids) <= set(known):
        raise ValueError("Unknown or duplicate case selection")
    cfg.setdefault("comparison_run", str(baseline))
    write(root/"config.json", cfg); write(root/"cases.json", ids)
    inputs = []
    for cid in ids:
        dest = root/"inputs"/cid; dest.mkdir(parents=True)
        source = baseline/"inputs"/cid/"source.png"
        shutil.copy2(source, dest/"source.png")
        with Image.open(source) as raw:
            native = raw.convert("RGB")
        view = global_frame(native.size)
        view.image(native).save(dest/"global.png")
        write(dest/"views.json", {"case_id": cid, "native_size": list(native.size),
            "source_sha256": sha256(dest/"source.png"),
            "views": [{"view_id": "G", "path": str((dest/"global.png").relative_to(root)),
                       "sha256": sha256(dest/"global.png"), "transform": view.metadata(native.size)}]})
        manifest = {"native_size": list(native.size), "global_size": list(view.input_size),
                    "global_transform": view.metadata(native.size), "local_views": []}
        for vid, frame in local_views(native.size, cfg["tile_overlap"]):
            frame.image(native).save(dest/(vid+".png"))
            manifest["local_views"].append({"id": vid, "path": vid+".png",
                "bounds_global": visible_global_bounds(frame, view),
                "transform": frame.metadata(native.size)})
        write(dest/"frames.json", manifest)
        inputs.append({"case_id": cid, "source": str(source), "sha256": sha256(source)})
        if cfg.get('frozen_global_from'):
            if cfg.get('replay_observations_from'):
                raise ValueError('Cannot replay all observations in a local discovery experiment')
            from frozen_global import freeze
            freeze(root, cid, Path(cfg['frozen_global_from']), cfg)
        if cfg.get('replay_observations_from'):
            replay_source = Path(cfg['replay_observations_from']) / 'cases' / cid / 'observations.json'
            replay_target = root / 'replay_observations' / (cid + '.json')
            replay_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(replay_source, replay_target)
    write(root/"input_provenance.json", inputs)
    files = {str(p.relative_to(root)): sha256(p) for sub in ("code", "inputs", "constraints_snapshot", "replay_observations", "frozen_global")
             for p in (root/sub).rglob("*") if p.is_file()}
    for n in ("config.json", "cases.json", "input_provenance.json"):
        files[n] = sha256(root/n)
    write(root/"manifest.json", {"created_at": now(), "files": files})
    publish(root)


def observed(answer, prefix):
    return [{**copy.deepcopy(r), "id": f"{prefix}_{r['id']}",
             "raw_model_id": r["id"], "source_stage": prefix}
            for r in answer["regions"]]


def run_case(root, cid, model, cfg):
    out = root/"cases"/cid; out.mkdir(parents=True, exist_ok=True)
    result_path = out/"result.json"
    if result_path.exists() and read(result_path).get("state") == "completed":
        return read(result_path)
    if result_path.exists():
        old = out/"resume_history"/now().replace(":", "-")
        old.mkdir(parents=True)
        for name in ("result.json", "reconstruction_handoff.json", "coordinate_audit.json", "status.json"):
            if (out/name).exists(): shutil.copy2(out/name, old/name)
    issues = []
    inp = root/"inputs"/cid; frames = read(inp/"frames.json")
    size = frames["global_size"]; view = global_frame(frames["native_size"])

    def state(stage):
        write(out/"status.json", {"state": "running", "stage": stage, "updated_at": now()})
        publish(root)

    def call(folder, prompt, images, validator, *, system=None):
        try:
            kwargs = {'system': system} if system is not None else {}
            result = model.ask(folder, prompt, images, validator, **kwargs)
            publish(root)
            return result
        except Exception as exc:
            detail = {"stage": str(folder.relative_to(out)), "type": type(exc).__name__, "error": str(exc)}
            write(folder/"unresolved.json", detail)
            return {"_failure": detail}

    replay = root / 'replay_observations' / (cid + '.json')
    if replay.exists():
        state('reused_discovery_for_local_review_ablation')
        observations = read(replay)['regions']
        C.regions_valid(observations, size)
        write(out / 'discovery_replay.json', {'frozen_source': str(replay), 'sha256': sha256(replay),
            'new_discovery_calls': False, 'new_local_review_calls': True})
        latest(root, cid, observations, '02_discovery_observations')
    else:
        state("frozen_global_and_new_local_discovery" if cfg.get('frozen_global_from') else "parallel_global_and_local_discovery")

        def global_job():
            if cfg.get('frozen_global_from'):
                from frozen_global import replay
                return 'G', replay(root, cid, size)
            return "G", call(out/"global_discovery", Draft.global_prompt(size),
                      [("GLOBAL clean — sole output coordinate frame", inp/"global.png")],
                      lambda a: C.discovery_valid(a, size), system=Draft.DISCOVERY_SYSTEM)

        def local_job(spec):
            folder = out/"local_discovery"/spec["id"]
            write(folder/"view.json", spec)
            q = Draft.local_prompt(size, spec["bounds_global"], spec["transform"]["actual_crop_input_size_px"])
            return spec["id"], call(folder, q,
                [("GLOBAL clean — sole output coordinate frame", inp/"global.png"),
                 ("LOCAL clean — magnified evidence only, never output this frame", inp/spec["path"])],
                lambda a: C.discovery_valid(a, size), system=Draft.DISCOVERY_SYSTEM)

        discovery_results = {}
        with ThreadPoolExecutor(max_workers=cfg["local_workers"]+1) as pool:
            jobs = [pool.submit(global_job)] + [pool.submit(local_job, s) for s in frames["local_views"]]
            for future in as_completed(jobs):
                vid, res = future.result()
                discovery_results[vid] = res
                if "_failure" in res: issues.append(res["_failure"])
                else:
                    rows = [r for v in sorted(discovery_results)
                            if "_failure" not in discovery_results[v]
                            for r in observed(discovery_results[v], v)]
                    latest(root, cid, rows, "02_discovery_observations")
        observations = [r for v in sorted(discovery_results)
                        if "_failure" not in discovery_results[v]
                        for r in observed(discovery_results[v], v)]
    write(out/"observations.json", {"coordinate_frame": "GLOBAL", "regions": observations})
    candidates = C.merge_near_identical(observations, cfg["observation_merge_iou"])
    write(out/"compressed_observations.json", {"regions": candidates,
        "rule": "Same representation and all-pairs bbox IoU >= threshold; no semantic deletion",
        "iou_threshold": cfg["observation_merge_iou"]})
    if not observations:
        # Empty valid model answers differ from total execution failure.
        if issues:
            raise RuntimeError("No usable observation calls: "+json.dumps(issues, ensure_ascii=False))
    from local_review import run as review_locally
    final, local_issues = review_locally(root, cid, candidates, model, cfg, state)
    issues.extend(local_issues)
    C.regions_valid(final, size)
    pending = sum(r["extent"] != "complete" for r in final)
    handoff, coords = export(cid, final, view, inp/"source.png",
                             defer_uncertain=cfg.get('review_defer_uncertain_to_missing', False))
    write(out/"reconstruction_handoff.json", handoff)
    write(out/"coordinate_audit.json", {"transform": view.metadata(frames["native_size"]), "regions": coords})
    objects = [{**r, "bbox_native": c["bbox_native"], "locator": r["label"]}
               for r, c in zip(final, coords)]
    result = {"case_id": cid, "state": "completed_with_issues" if issues or pending else "completed",
              "regions": final, "objects": objects, "region_count": len(final),
              "observations_count": len(observations), "compressed_count": len(candidates),
              "issues": issues, "pending_count": pending, "completed_at": now(),
              "initial_geometry_task_count": len(handoff['tasks']),
              "deferred_missing_candidate_count": len(handoff['deferred_candidates']),
              "geometry_executed": False, "formal_accuracy_measured": False}
    write(result_path, result)
    write(out/"finished.json", {"completed_at": result["completed_at"], "state": result["state"],
                                "pending_count": pending, "issue_count": len(issues)})
    write(out/"status.json", {"state": result["state"], "stage": "finished", "updated_at": now()})
    latest(root, cid, final, "05_final")
    return result


def run(root, selected=None):
    if Path(__file__).resolve().parent != root/"code":
        cmd = [sys.executable, "-B", "-u", str(root/"code/run.py"), "run", "--run-dir", str(root)]
        if selected: cmd += ["--cases", *selected]
        os.execv(sys.executable, cmd)
    lock = (root/"run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in read(root/"manifest.json")["files"].items():
        if sha256(root/name) != digest:
            raise RuntimeError("Frozen file changed: "+name)
    cfg = read(root/"config.json")
    all_ids = read(root/"cases.json"); ids = selected or all_ids
    if not set(ids) <= set(all_ids):
        raise ValueError("Unknown case")
    model = Model(cfg, run_dir=root, system=Draft.DISCOVERY_SYSTEM)
    write(root/"process.json", {"pid": os.getpid(), "started_at": now(), "selected_cases": ids})
    stopping = threading.Event()

    def periodic():
        while not stopping.wait(10):
            try: publish(root)
            except Exception: traceback.print_exc()

    watcher = threading.Thread(target=periodic, daemon=True); watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=cfg["case_workers"]) as pool:
            jobs = {pool.submit(run_case, root, cid, model, cfg): cid for cid in ids}
            for future in as_completed(jobs):
                cid = jobs[future]
                try:
                    r = future.result()
                    print(cid, r["state"], "regions", r["region_count"], flush=True)
                except Exception as exc:
                    write(root/"cases"/cid/"failed.json", {"error": str(exc), "traceback": traceback.format_exc()})
                    write(root/"cases"/cid/"status.json", {"state": "failed", "stage": "failed", "error": str(exc), "updated_at": now()})
                    print(cid, "failed", str(exc), flush=True)
                publish(root)
    finally:
        stopping.set(); watcher.join()
        write(root/"execution.json", {"state": "finished", "finished_at": now(), "selected_cases": ids})
        publish(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args(); root = args.run_dir.resolve()
    if args.action == "prepare":
        prepare(root, args.baseline.resolve(), Path(__file__).parent, args.config, args.constraints, args.cases)
    elif args.action == "report": publish(root)
    else: run(root, args.cases)
