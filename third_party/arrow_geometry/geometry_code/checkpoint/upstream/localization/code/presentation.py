"""Lossless, relative-link evidence and a reloadable read-only report."""
import html
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transport.common import read_json as read, write_json as write

LOCK = threading.RLock()
COLORS = {"linework": "#2877cf", "block_arrow": "#ec8100"}


def now():
    return datetime.now(timezone.utc).isoformat()


def overlay(source, regions, destination):
    im = Image.open(source).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    occupied = []
    for r in regions:
        box = tuple(r["bbox"])
        color = COLORS[r["render_type"]] if r.get("extent") != "uncertain" else "#cb3535"
        d.rectangle(box, outline=color, width=1)
        label = r["id"]
        size = d.textbbox((0, 0), label, font=font)
        tw, th = size[2]-size[0]+4, size[3]-size[1]+4
        choices = [(box[0], box[1]-th-1), (box[2]-tw, box[3]+1),
                   (box[0]-tw-1, box[1]), (box[2]+1, box[1])]
        def clip(x, y):
            return max(0, min(im.width-tw, x)), max(0, min(im.height-th, y))
        positions = [clip(x, y) for x, y in choices]
        def collisions(pos):
            x, y = pos
            return sum(max(0, min(x+tw, b[2])-max(x, b[0])) *
                       max(0, min(y+th, b[3])-max(y, b[1])) for b in occupied)
        x, y = min(positions, key=collisions)
        rect = (x, y, x+tw, y+th)
        d.line((box[0], box[1], x+tw/2, y+th/2), fill=color, width=1)
        d.rectangle(rect, fill="white", outline=color)
        d.text((x+2, y-size[1]+2), label, fill=color, font=font)
        occupied.append(rect)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp.png")
    im.save(tmp)
    tmp.replace(destination)


def compact(rows):
    return [{k: r[k] for k in ("id", "bbox", "render_type", "grouping", "label", "extent")} for r in rows]


def publish(root):
    """No model calls. Assets are copied once or versioned by stage mtime."""
    with LOCK:
        cfg = read(root/"config.json")
        ids = read(root/"cases.json")
        dest = root/"reports"; assets = dest/"assets"
        assets.mkdir(parents=True, exist_ok=True)
        records = []; fragments = []
        md = [f"# 定位重构：{len(ids)} 图实时对照", "", f"更新：{now()}",
              "", "蓝框：线式箭头/独立线条；橙框：块状箭头；红框：模型仍不确定。",
              "这是定位范围，不是几何重建结果。完成数和框数不代表召回率或分组正确率。",
              "", "[自动刷新的网页](live.html) · [仅全局发现对照](24图_全局发现对照.md) · [运行配置](../config.json)", ""]
        baseline = Path(cfg["comparison_run"])
        baseline_title = cfg.get("comparison_title", "v29 对照")
        for cid in ids:
            c = root/"cases"/cid
            state = read(c/"status.json") if (c/"status.json").exists() else {"state": "queued", "stage": "queued"}
            result = read(c/"result.json") if (c/"result.json").exists() else {}
            latest = read(c/"latest.json") if (c/"latest.json").exists() else {}
            base_source = baseline/"cases"/cid/"representation.png"
            if not base_source.exists():
                base_source = baseline/"cases"/cid/"overlay.png"
            if not base_source.exists():
                base_source = baseline/"cases"/cid/"previews"/"05_final.png"
            original_asset = assets/(cid+"_source.png")
            if not original_asset.exists():
                shutil.copy2(root/"inputs"/cid/"source.png", original_asset)
            baseline_stamp = str(base_source.stat().st_mtime_ns) if base_source.exists() else "pending"
            baseline_asset = assets/(cid+"_baseline_"+baseline_stamp+".png")
            if base_source.exists() and not baseline_asset.exists():
                shutil.copy2(base_source, baseline_asset)
            current_asset = None
            if latest.get("overlay") and (root/latest["overlay"]).exists():
                source = root/latest["overlay"]
                current_asset = assets/(cid+"_"+str(source.stat().st_mtime_ns)+".png")
                if not current_asset.exists():
                    shutil.copy2(source, current_asset)
            label = state.get("state", "")+" / "+state.get("stage", "")
            counts = result.get("region_count")
            record = {"case_id": cid, **state, "region_count": counts,
                      "issues": result.get("issues", []), "pending": result.get("pending_count", 0)}
            records.append(record)
            md += ["## "+cid, "", label + (f"；最终区域 {counts}；待确认 {record['pending']}" if counts is not None else ""),
                   "", f"[原图](assets/{original_asset.name}) · [Trace](../cases/{cid}/) · [交接](../cases/{cid}/reconstruction_handoff.json)", "",
                   f"| {baseline_title} | 本次最新定位 |", "| --- | --- |",
                   "| "+(f"[![对照](assets/{baseline_asset.name})](assets/{baseline_asset.name})" if baseline_asset.exists() else "对照尚未结束")+
                   " | "+(f"[![本次](assets/{current_asset.name})](assets/{current_asset.name})" if current_asset else "等待") + " |", ""]
            if record["issues"]:
                md += ["执行问题："+json.dumps(record["issues"], ensure_ascii=False), ""]
            base_html = f'<a href="assets/{baseline_asset.name}"><img src="assets/{baseline_asset.name}"></a>' if baseline_asset.exists() else "无对照"
            current_html = f'<a href="assets/{current_asset.name}"><img src="assets/{current_asset.name}"></a>' if current_asset else "等待"
            fragments.append(f'<section id="{cid}"><h2>{cid} · {html.escape(label)}</h2><a href="assets/{original_asset.name}">原图</a> · <a href="../cases/{cid}/">Trace</a><div class="pair"><div>{html.escape(baseline_title)}{base_html}</div><div>本次{current_html}</div></div></section>')
        finished = sum(r["state"] in ("completed", "completed_with_issues", "failed") for r in records)
        progress = {"updated_at": now(), "finished": finished, "total": len(ids),
                    "running": sum(r["state"] == "running" for r in records),
                    "queued": sum(r["state"] == "queued" for r in records),
                    "cases": records}
        write(root/"progress.json", progress)
        (dest/"24图_实时对照.md").write_text("\n".join(md)+"\n")
        page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>24 图定位实验</title><style>body{font:16px system-ui;margin:24px;background:#f5f7fb}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{display:block;width:100%;background:white}section{margin:30px 0;background:white;padding:16px}a{color:#1767bb}</style>'
        page += f"<h1>定位实验 {finished}/{len(ids)}</h1><p>每20秒刷新。{now()} · 只显示定位范围；框数不代表准确率。</p><p><a href='global_live.html'>只看全局发现的新旧对照</a></p>"
        if cfg.get('frozen_global_from'):
            page += '<p><a href="local_live.html">局部发现规则对齐前后（审核前）</a></p>'
        page += " · ".join(f'<a href="#{cid}">{cid}</a>' for cid in ids)
        (dest/"live.html").write_text(page+"".join(fragments))
        (root/"RESULTS.md").write_text("# 模型主导定位实验\n\n"+
             f"当前结束 {finished}/{len(ids)}，运行 {progress['running']}，排队 {progress['queued']}。\n\n"+
             "[24图两栏对照](reports/24图_实时对照.md) · [自动刷新](reports/live.html) · [配置](config.json) · [实现来源](code/reuse_provenance.json)\n\n"+
             f"默认思考（不传reasoning_effort），{cfg['max_tokens']}输出上限，{cfg['case_workers']}图/API{cfg['api_concurrency']}。"+
             ("固定v015全局结果，2×2局部重新调用（仅规则对齐，不输入全局候选）" if cfg.get('frozen_global_from') else
              "固定历史发现候选（只用于本次修正消融）" if cfg.get('replay_observations_from') else "全局与2×2局部独立并行发现/补漏")+
             "→仅高IoU观察压缩→小批局部bbox校正→跨批局部交接复查→最终交接；无全局修正。"+
             "发现与最终交接使用GLOBAL像素；局部修正只输出LOCAL像素，后端记录精确映射及边界交集。\n\n"+
             ("每张定位结束后自动接几何恢复和审核：[定位与重建总览](reports/本轮定位与重建对照.md)。\n" if cfg.get('reconstruction_enabled') else
              "本轮只验证定位与交接；没有执行下游几何/PPT或块状轮廓恢复。")+
              ("\n\n本轮整理输入删除所有候选 label 字段；保留 A 的类型继承。框、grouping、extent、图片、分组规则和轮次不变；模型仍可生成新的输出描述。旧候选若 keep，后台仍保留旧描述。详见 [实验设置](EXPERIMENT_SETUP.md)。" if cfg.get('review_omit_candidate_labels', False) else "")+
              "视觉质量仍需逐例核查，不能由完成状态判断。\n\n[本轮实际提示词](PROMPTS.md) · [全局发现单独对照](reports/24图_全局发现对照.md)\n"+
              ("\n[局部发现规则对齐前后](reports/24图_局部发现对照.md) · [局部自动刷新](reports/local_live.html)\n" if cfg.get('frozen_global_from') else ""))
        from global_report import publish_global
        publish_global(root, overlay)
        from local_report import publish_local
        publish_local(root, overlay, now())


def latest(root, cid, rows, stage):
    c = root/"cases"/cid
    dest = c/"previews"/(stage+".png")
    overlay(root/"inputs"/cid/"global.png", rows, dest)
    write(c/"latest.json", {"stage": stage, "overlay": str(dest.relative_to(root)), "updated_at": now()})
    publish(root)
