"""Structural layout checks, independent of image scores and font heuristics."""
from copy import deepcopy
import unicodedata

from ..common import intersect


def overlap(a, b):
    area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
    return intersect(a,b)/max(1e-9,min(area(a),area(b)))


def new_conflicts(rows, updates, threshold=.20):
    """Propose joint recovery BEFORE rolling good updates back to bad old boxes."""
    baseline={r['id']:r for r in rows}
    active=[(r['id'],child) for r in rows for child in updates.get(r['id'],[r])]
    events=[]
    for index,(pa,a) in enumerate(active):
        for pb,b in active[index+1:]:
            now=overlap(a['bbox_2d'],b['bbox_2d'])
            if now<threshold:continue
            if pa==pb and (a.get('kind')=='formula' or b.get('kind')=='formula'):continue
            previous=0. if pa==pb else overlap(baseline[pa]['bbox_2d'],baseline[pb]['bbox_2d'])
            if previous>=.10 or now-previous<.15:continue
            events.append({'parents':list(dict.fromkeys((pa,pb))),
                           'ids':[a['id'],b['id']], 'overlap_smaller_fraction':now,
                           'baseline_overlap':previous})
    return events


def attribute_layout(unit, kind):
    """Separate glyph orientation from writing direction and baseline count."""
    count=unit.get('visual_line_count',1)
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=64:
        raise ValueError('visual_line_count must be an integer in 1..64')
    mode=unit.get('layout_mode','multiline' if count>1 else 'single_line')
    if mode not in ('single_line','multiline','stacked_vertical'):
        raise ValueError('Unknown layout_mode')
    if kind=='formula':
        if mode=='stacked_vertical':raise ValueError('A formula cannot use stacked plain glyphs')
        return 'single_line',1
    if mode=='stacked_vertical':
        if abs(float(unit.get('rotation_deg',0)))>1e-6:
            raise ValueError('stacked_vertical requires upright glyphs, rotation_deg=0')
        return mode,1
    if mode=='multiline' or count>1:
        if count<2:raise ValueError('multiline requires the observed baseline count')
        return 'multiline',count
    return mode,1


def stacked_runs(row):
    """One upright non-whitespace glyph per line; original row/runs stay intact."""
    glyphs=[]
    for run in row['runs']:
        for char in run['text']:
            if char.isspace():continue
            if glyphs and (unicodedata.combining(char) or char in ('\ufe0e','\ufe0f')):
                glyphs[-1].append({**run,'text':char})
            else:glyphs.append([{**run,'text':char}])
    return glyphs


def reconcile(rows, updates, threshold=.20):
    """Roll back whole parents that acquire NEW geometric conflicts.

    Retain all original occurrences. Existing overlap (e.g. formula scripts) is
    not a reason to delete/merge text. Recheck after rollback because retained
    baselines can reveal a conflict with a different, newly moved candidate.
    """
    baseline={r['id']:r for r in rows};kept=deepcopy(updates);warnings=[]
    for _ in range(len(updates)+1):
        active=[(r['id'],child) for r in rows for child in kept.get(r['id'],[r])]
        rejected=set();evidence=[]
        for i,(pa,a) in enumerate(active):
            for pb,b in active[i+1:]:
                now=overlap(a['bbox_2d'],b['bbox_2d'])
                if now<threshold:continue
                if pa==pb:
                    # New children of one plain-text parent need distinct lines.
                    if a.get('kind')=='formula' or b.get('kind')=='formula':continue
                    previous=0.
                else:
                    previous=overlap(baseline[pa]['bbox_2d'],baseline[pb]['bbox_2d'])
                if previous>=.10 or now-previous<.15:continue
                parents={p for p in (pa,pb) if p in kept}
                if parents:
                    rejected.update(parents)
                    evidence.append({'ids':[a['id'],b['id']],
                                     'overlap_smaller_fraction':now,'baseline_overlap':previous})
        if not rejected:break
        for parent in sorted(rejected):
            kept.pop(parent)
            warnings.append({'stage':'layout_conflict','id':parent,
                'reason':'New text-box collision; original parent geometry/content/styles retained',
                'conflicts':[e for e in evidence if any(
                    ident==parent or ident.startswith(parent+'.') for ident in e['ids'])]})
    return kept,warnings
