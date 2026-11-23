"""A localization-scope filter, not a claim about recovered geometry."""
import copy
from pathlib import Path
import numpy as np
from PIL import Image


def contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def filter_candidates(groups, baseline, coverage_threshold=1.0):
    if not 0 < coverage_threshold <= 1:
        raise ValueError('Candidate coverage threshold must be in (0,1]')
    retained=[];excluded=[]
    for group in groups:
        a=group['bbox']; area=max(0,a[2]-a[0])*max(0,a[3]-a[1])
        matches=[]
        for row in baseline:
            b=row['bbox']; overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            fraction=overlap/area if area else 0
            if (contains(b,a) if coverage_threshold==1.0 else fraction>=coverage_threshold):
                matches.append(dict(id=row['id'],candidate_coverage=fraction))
        covering=[m['id'] for m in matches]
        if covering:
            excluded.append(dict(id=group['id'],bbox=copy.deepcopy(group['bbox']),
                member_ids=list(group['member_ids']),contained_by=covering,coverage_matches=matches,
                reason='SAM candidate area is covered by one existing localization scope'))
        else:retained.append(copy.deepcopy(group))
    seen=[x['id'] for x in retained+excluded]
    if sorted(seen)!=sorted(x['id'] for x in groups) or len(set(seen))!=len(seen):
        raise ValueError('Candidate partition is not lossless')
    return retained,excluded,dict(compressed_count=len(groups),excluded_count=len(excluded),
        retained_count=len(retained),excluded_observations=sum(len(x['member_ids']) for x in excluded),
        retained_observations=sum(len(x['member_ids']) for x in retained),
        baseline_modified=False,original_evidence_deleted=False,geometry_coverage_verified=False,
        coverage_threshold=coverage_threshold,
        rule='Intersection / SAM candidate bbox area by one existing bbox; no union of old boxes. Threshold 1.0 preserves exact containment.',
        coordinate_policy='Use saved native-coordinate SAM pixel footprints and saved model bbox corners directly, without rounding, tolerance or extra coordinate shifts.')


def retained_mask_union(raw, groups, frames, native_size):
    """Include whole masks of retained groups; do not clip retained branches to old boxes."""
    ids={member for group in groups for member in group['member_ids']}
    known={row['id'] for row in raw}
    if not ids<=known:raise ValueError('Unknown mask observation')
    result=np.zeros((native_size[1],native_size[0]),dtype=np.uint8)
    for row in raw:
        if row['id'] not in ids:continue
        frame=frames[row['view']]
        with Image.open(row['mask_path']) as im:
            if im.size!=frame.input_size:raise ValueError('Mask/frame mismatch')
            mask=np.asarray(im.convert('L'))>127
        raster=np.asarray(Image.fromarray(mask.astype('uint8')*255).resize(frame.size,Image.Resampling.NEAREST))
        x0,y0,x1,y1=frame.box
        np.maximum(result[y0:y1,x0:x1],raster,out=result[y0:y1,x0:x1])
    return Image.fromarray(result)


def comparison_report(root,read,now):
    cfg=read(root/'config.json');a=root.parent/cfg['control_run_name']
    lines=['# SAM包含候选过滤：五图对照','',f'更新 {now()}。A为v025，B为本轮v027。原定位、原图、SAM扫描固定；不执行几何。',
        '完整落在一个已有框里的SAM区域从补漏候选及弱高亮中排除，框外/伸出的区域保留。原始证据文件完整保留。',
        '当前只检查定位范围之外的遗漏，不对已有框内部重新做一遍发现。候选数下降不代表召回改善；需要看原图与最终定位。','',
        '| case | 原候选 | 包含排除 | 剩余 | A进度 | B进度 |','|---|---:|---:|---:|---|---|']
    details=[]
    for cid in read(root/'cases.json'):
        audit=read(root/'cases'/cid/'filter_audit.json',{})
        sa=read(a/'cases'/cid/'status.json',{});sb=read(root/'cases'/cid/'status.json',{})
        lines.append(f"| {cid} | {audit.get('compressed_count','—')} | {audit.get('excluded_count','—')} | {audit.get('retained_count','—')} | {sa.get('state','等待')}/{sa.get('stage','')} | {sb.get('state','等待')}/{sb.get('stage','')} |")
        details += [f'## {cid}','',f'[原图](../inputs/{cid}/source.png) · [原定位](../cases/{cid}/baseline.png) · [排除清单](../cases/{cid}/excluded_candidates.json) · [输入前后对照](../preflight/{cid}/README.md) · [B trace](../cases/{cid}/)','',
            '| A：未做包含过滤 | B：过滤后补漏 |','|---|---|',
            f'| [![A](../../{a.name}/cases/{cid}/current.png)](../../{a.name}/cases/{cid}/current.png) | [![B](../cases/{cid}/current.png)](../cases/{cid}/current.png) |','']
    lines+=['','[实验说明](../README.md) · [输入数量与字符统计](../preflight/input_metrics.json) · [结果记录](../OBSERVATIONS.md)','']+details
    dest=root/'reports/包含过滤_五图对照.md';tmp=dest.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(dest)
