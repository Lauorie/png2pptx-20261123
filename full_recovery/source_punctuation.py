"""Retire an isolated glyph only when surviving native ink replaces it.

The source and rendered components must agree in position and extent, including
the gaps between them. A continuous shaft, geometric overlap or a class label
cannot witness deletion. No arrow product or source geometry is changed.
"""
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def candidates(rows):
    import unicodedata
    def standalone(row):
        value=''.join(row.get('text','').split())
        # Repeated standalone symbols can be geometric marks just like one
        # glyph. Spelling only nominates; complete source/native ink still
        # proves each component before any text is retired.
        repeated=(1<=len(value)<=8 and len(set(value))==1 and
                  unicodedata.category(value[0])[0] in ('P','S'))
        return len(value)==1 or repeated or value in ('...', '…', '⋯', '⋮', '⋱', '⋰')
    return [r for r in rows if r.get('kind','text')!='formula' and
            standalone(r)]


def components(rgb, pigment):
    mask = (np.linalg.norm(rgb.astype(float)-pigment, axis=2) < 70).astype('uint8')
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for index, ((x, y, w, h, area), center) in enumerate(zip(stats[1:], centers[1:]), 1):
        # An enclosing node/border cut by the view is context, not a complete
        # replacement witness. Small source dashes need not be round dots.
        if x == 0 or y == 0 or x+w == rgb.shape[1] or y+h == rgb.shape[0]:
            continue
        result.append({'center': center, 'extent': np.array([w, h]),
                       'area': int(area), 'mask': labels == index})
    return result


def replacement_refs(points, pigment, scene, style_data, arrows, plan):
    from .handoff import arrow_strokes, distances, surviving_scene
    from .text_occlusion import part_strokes
    table = [[] for _ in points]
    def same_color(color):
        return (isinstance(color, str) and len(color) == 7 and
                np.linalg.norm(np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])-pigment) < 80)
    for key, part in surviving_scene(scene, plan)['parts'].items():
        if part.get('status') != 'ok':
            continue
        style = style_data.get('parts', {}).get(key, {}).get('style', {})
        suppressed = plan.get('shape_segment_suppression', {}).get(
            part['object_id']+'/'+part.get('id', key.split('/')[-1]), [])
        part = {**part, '_composition_suppressed_segments': suppressed}
        paths = part_strokes(part)
        covered = np.zeros(len(points), bool)
        if same_color(style.get('stroke')) and paths:
            covered |= distances(points, paths) <= max(1.5, (style.get('width_px') or 1)/2+1)
        if same_color(style.get('fill')):
            for path in paths:
                if len(path) >= 4 and np.linalg.norm(path[0]-path[-1]) < 1e-6:
                    contour = np.asarray(path, 'float32')
                    covered |= np.array([cv2.pointPolygonTest(contour, tuple(map(float, p)), True) >= -1.5
                                         for p in points])
        for i in np.flatnonzero(covered):
            table[i].append('shape:'+key)
    for arrow in arrows:
        if arrow['id'] in set(plan.get('drop_arrow_names', [])):
            continue
        style = arrow.get('style', {})
        if not same_color(style.get('color') or style.get('stroke')):
            continue
        paths = arrow_strokes(arrow)
        if not paths:
            continue
        covered = distances(points, paths) <= max(1.5, (style.get('width_px') or 1)/2+1)
        for i in np.flatnonzero(covered):
            table[i].append('arrow:'+arrow['id'])
    return table


def observe(source, native, rows, scene, styles, arrow_scene, plan, glyph_native=None):
    """Test repeated symbol paint against named local native evidence.

    OCR can misread a standalone mark's shade or glyph. Try the existing
    overlapping primitive's pigment as an identity hypothesis.
    Source/native component position, count, ink and ownership still decide.
    """
    from shapely.geometry import box
    selected_candidates=candidates(rows)
    if not selected_candidates:return list(rows),[]
    pixels=(np.asarray(source.convert('RGB')),np.asarray(native.convert('RGB')),
            np.asarray(glyph_native.convert('RGB')) if glyph_native is not None else None)
    result=[];receipts=[]
    for row in selected_candidates:
        value=''.join(row.get('text','').split())
        hypotheses=[row]
        if (len(value)>1 and len(set(value))==1) or value in ('...', '…', '⋯', '⋮', '⋱', '⋰'):
            b=np.asarray(row['bbox_2d'],float);region=box(*b).buffer(max(2.,min(b[2:]-b[:2])/2))
            colors=set()
            for key,part in scene.get('parts',{}).items():
                if key in set(plan.get('drop_shape_keys',())) or part.get('status')!='ok':continue
                pb=part.get('bbox_2d')
                if not pb or not region.intersects(box(*pb)):continue
                paint=styles.get('parts',{}).get(key,{}).get('style',{})
                # Foreground contours nominate a pigment; enclosed background
                # holes cannot substitute for the text occurrence's ink.
                color=paint.get('stroke')
                if isinstance(color,str) and color!=row.get('color'):colors.add(color)
            hypotheses.extend({**row,'color':color} for color in sorted(colors)[:8])
        tested=[_observe(source,native,[hypothesis],scene,styles,arrow_scene,plan,glyph_native,pixels=pixels)
                for hypothesis in hypotheses]
        proofs=[audit[0] for kept,audit in tested if not kept and audit]
        if proofs:
            receipt=proofs[0];receipt['source_pigment_hypothesis']=hypotheses[
                next(i for i,(kept,audit) in enumerate(tested) if not kept)]['color']
        else:
            result.append(row);receipt=tested[0][1][0]
        receipts.append(receipt)
    selected={r['id'] for r in result};considered={r['id'] for r in selected_candidates}
    return [r for r in rows if r['id'] not in considered or r['id'] in selected],receipts


def _observe(source, native, rows, scene, styles, arrow_scene, plan, glyph_native=None, *, pixels=None):
    if source.size != native.size:
        raise ValueError('Punctuation witnesses require the same source/native canvas')
    if glyph_native is not None and glyph_native.size!=source.size:
        raise ValueError('Glyph arbitration requires the same source canvas')
    rgb,rendered,glyph_pixels=(pixels if pixels is not None else
        (np.asarray(source.convert('RGB')),np.asarray(native.convert('RGB')),
         np.asarray(glyph_native.convert('RGB')) if glyph_native is not None else None))
    retired = set(); receipts = []
    for row in candidates(rows):
        record = {'text_id': row['id'], 'action': 'preserve', 'replacement_refs': []}
        receipts.append(record)
        color = row.get('color')
        if not isinstance(color, str) or len(color) != 7:
            record['reason'] = 'no_source_pigment'; continue
        pigment = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
        box = np.asarray(row['bbox_2d'], float); span = box[2:]-box[:2]
        if min(span) <= 0:
            record['reason'] = 'invalid_source_neighborhood'; continue
        margin = max(2., min(span)/2, .25*max(span))
        lo = np.maximum(0, np.floor(box[:2]-margin)).astype(int)
        hi = np.minimum(rgb.shape[1::-1], np.ceil(box[2:]+margin)).astype(int)
        if np.any(hi <= lo):
            record['reason'] = 'empty_source_neighborhood'; continue
        source_parts = components(rgb[lo[1]:hi[1], lo[0]:hi[0]], pigment)
        native_parts = components(rendered[lo[1]:hi[1], lo[0]:hi[0]], pigment)
        # Padding exposes context; it must not turn a neighboring label into
        # another component of this occurrence. Keep complete components whose
        # centers lie in its nominated ink box. No contour is clipped, and each
        # retained source component still requires a surviving native witness.
        def in_occurrence(part):
            center = part['center'] + lo
            return np.all(center >= box[:2]-1) and np.all(center <= box[2:]+1)
        source_parts = [part for part in source_parts if in_occurrence(part)]
        native_parts = [part for part in native_parts if in_occurrence(part)]
        record.update(source_component_count=len(source_parts), native_component_count=len(native_parts),
                      source_box_half_open=[*lo.tolist(), *hi.tolist()])
        if not 1 <= len(source_parts) <= 8 or len(source_parts) != len(native_parts):
            record['reason'] = 'source_and_native_component_counts_do_not_agree'; continue
        samples=[];owners=[]
        for index,part in enumerate(source_parts):
            yy,xx=np.where(part['mask']);stride=max(1,int(np.ceil(len(xx)/16)))
            points=np.column_stack((xx[::stride],yy[::stride]))+lo
            samples.extend(points);owners.extend([index]*len(points))
        sample_refs = replacement_refs(np.asarray(samples), pigment, scene, styles, arrow_scene['arrows'], plan)
        owner_support = [sum(bool(ref) for owner, ref in zip(owners, sample_refs) if owner == i)
                         / sum(owner == i for owner in owners) for i in range(len(source_parts))]
        # Native fitting may shift a thin contour slightly. The same .9
        # bidirectional ink support required below bounds this nomination;
        # every component still needs a named surviving replacement.
        record['native_owner_support'] = owner_support
        if min(owner_support) < .9:
            record['reason'] = 'no_surviving_native_reference_for_every_component'; continue
        refs=[sorted({ref for owner,group in zip(owners,sample_refs) if owner==i for ref in group})
              for i in range(len(source_parts))]
        cost = np.linalg.norm(np.array([p['center'] for p in source_parts])[:, None]-
                              np.array([p['center'] for p in native_parts])[None], axis=2)
        si, ni = linear_sum_assignment(cost)
        evidence = []; valid = True
        for i, j in zip(si, ni):
            a, b = source_parts[i], native_parts[j]
            tolerance = max(1., min(2., .3*float(a['extent'].max())))
            extents_match = np.all(np.abs(a['extent']-b['extent']) <= np.maximum(1., .35*a['extent']))
            area_ratio = b['area']/a['area']
            # Bidirectional one-pixel support prevents a hollow ring or a
            # different component shape from borrowing the same bounding box.
            kernel = np.ones((3, 3), 'uint8')
            support = float(cv2.dilate(b['mask'].astype('uint8'), kernel)[a['mask']].mean())
            reverse = float(cv2.dilate(a['mask'].astype('uint8'), kernel)[b['mask']].mean())
            valid &= (cost[i, j] <= tolerance and extents_match and .5 <= area_ratio <= 2.
                      and min(support, reverse) >= .9)
            evidence.append({'source_center': (a['center']+lo).tolist(),
                'native_center': (b['center']+lo).tolist(), 'source_extent': a['extent'].tolist(),
                'native_extent': b['extent'].tolist(), 'area_ratio': area_ratio,
                'source_support': support, 'native_support': reverse, 'replacement_refs': refs[i]})
        record['components'] = evidence
        if not valid and glyph_pixels is not None and min(e['source_support'] for e in evidence)>=.9:
            # An isolated, named native occurrence can fit imperfectly and
            # still dominate a misplaced glyph in BOTH source recall and
            # source precision. Its fit error cannot authorize adding a second
            # representation of that source occurrence.
            glyph_parts=components(glyph_pixels[lo[1]:hi[1],lo[0]:hi[0]],pigment)
            if len(glyph_parts)==len(source_parts):
                original=np.logical_or.reduce([p['mask'] for p in source_parts])
                native_mask=np.logical_or.reduce([p['mask'] for p in native_parts])
                glyph_mask=np.logical_or.reduce([p['mask'] for p in glyph_parts])
                kernel=np.ones((3,3),'uint8')
                def agreement(mask):
                    recall=float(cv2.dilate(mask.astype('uint8'),kernel)[original].mean())
                    precision=float(cv2.dilate(original.astype('uint8'),kernel)[mask].mean())
                    return recall,precision
                native_score=agreement(native_mask);glyph_score=agreement(glyph_mask)
                dominates=all(a>=b for a,b in zip(native_score,glyph_score)) and any(
                    a>b+1e-6 for a,b in zip(native_score,glyph_score))
                record['rendered_source_arbitration']={
                    'native_recall':native_score[0],'native_precision':native_score[1],
                    'glyph_recall':glyph_score[0],'glyph_precision':glyph_score[1],
                    'native_dominates':dominates,'physical_component_count':len(source_parts),
                    'native_fit_still_unresolved':True}
                if dominates:valid=True
        if not valid:
            record['reason'] = 'rendered_native_ink_does_not_replace_source_components'; continue
        record.update(action='retire_duplicate_text',
            reason='Surviving native references already render these original visible components and gaps',
            replacement_refs=sorted({ref for group in refs for ref in group}))
        retired.add(row['id'])
    return [deepcopy(r) for r in rows if r['id'] not in retired], receipts
