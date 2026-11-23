"""Fixed SAM evidence schedule with explicit candidate accounting.

Regional discovery and final coverage use their existing model calls. Each
displayed proposal receives a disposition; a proposal never becomes geometry
without source-image confirmation. Object checks share the normal revision
budget and do not introduce another reconstruction loop.
"""
import copy
from pathlib import Path

from .common import read, save, mapped_box
from .sam_guidance import GuidedSamEvidence
from .sam_evidence import iou, area, overlap, local_box


DECISION_SCHEMA = {
    'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['sam_id', 'action', 'target', 'reason'],
        'properties': {
            'sam_id': {'type': 'string'},
            'action': {'enum': ['covered', 'add', 'reject', 'uncertain']},
            'target': {'type': ['string', 'integer', 'null']},
            'reason': {'type': 'string', 'minLength': 1}}}}
ACCOUNTING_RULE = (
    'Account for EVERY listed SAM ID exactly once in sam_decisions. covered: '
    'the same object or intrinsic part is represented by an existing candidate '
    'or a parent in this response new_objects; target is the existing ID or the '
    'zero-based new_objects index (a new local ID is also valid). add: the clean image confirms a separate missing '
    'object; include it in new_objects and set target to its zero-based index. '
    'reject: not an in-scope object or a false proposal; target=null. uncertain: '
    'identity/extent cannot be resolved here; target=null. Give a short visual '
    'reason. Several SAM proposals may refer to one object. An enclosing panel '
    'is distinct from its contents, including photos. Its own fill boundary can '
    'define a shape without a stroke. Containment alone is not duplicate identity. '
    'SAM is fallible; do not add an outline solely because a proposal has a box.')


class FixedSamEvidence(GuidedSamEvidence):
    def __init__(self, *args, precheck=True, **kwargs):
        kwargs.update(activation='all')
        super().__init__(*args, **kwargs)
        self.precheck_enabled = precheck
        self.clean_fit_view = True

    def _accounting(self, prompt, groups, *, formal):
        if not groups:
            return prompt
        prompt = copy.deepcopy(prompt)
        prompt['sam_candidate_accounting'] = ACCOUNTING_RULE
        if formal:
            prompt['output_schema']['properties']['sam_decisions'] = DECISION_SCHEMA
            prompt['output_schema']['required'].append('sam_decisions')
        else:
            prompt['output_schema']['sam_decisions'] = [
                {'sam_id': 'listed SAM ID', 'action': 'covered|add|reject|uncertain',
                 'target': 'existing ID, new_objects index, or null', 'reason': 'source evidence'}]
            prompt['serialization'] = prompt['serialization'].replace(
                'exactly the four listed root keys', 'exactly the five listed root keys')
        return prompt

    def region(self, prompt, images, frame, assignment, folder):
        work = mapped_box(assignment['region']['bbox_2d'], frame['global_image_size_px'],
                          frame['original_image_size_px'])
        groups, count = self._unrepresented(work, assignment['candidates'])
        path = Path(folder)/'sam_candidates.png'
        records = self.render(images[0][1], groups, frame['source_crop_half_open'], path, masks=False) if groups else []
        save(Path(folder)/'sam_evidence.json', {'stage': 'regional_review', 'shown': records,
                                             'eligible': count, 'truncated': count > len(records)})
        if not records:
            return prompt, images
        prompt = copy.deepcopy(prompt)
        boxes = {o['id']: local_box(o['bbox_2d'], frame['source_crop_half_open'],
                                   frame['actual_crop_input_size_px']) for o in assignment['candidates']}
        for candidate in prompt['candidates']:
            candidate['identity_bbox_local'] = [round(v, 1) for v in boxes[candidate['id']]]
        prompt['input_images'] = prompt['input_images'].replace('Only visual old boxes are supplied.',
            'Listed boxes identify objects in the same crop coordinates; context boxes can extend outside '
            'the crop. They are localization hints, not fitted contours.')
        prompt['sam_auxiliary_evidence'] = {
            'candidates': [{'id': r['id'], 'category_hint': r['category'],
                            'bbox_local': [round(v, 1) for v in r['bbox_local']]} for r in records],
            'rule': 'These are unrepresented location proposals, not confirmed shapes. '
                    'Inspect the clean crop and account for each; absence from SAM never rejects an existing object.'}
        return self._accounting(prompt, records, formal=False), [
            *images, ('Unrepresented SAM proposals aligned to the clean crop', path)]

    def _unrepresented(self, region, existing):
        rows = []
        for group in self.groups:
            p = group['representative']; b = p['bbox']
            if not self.eligible(group) or p['score'] < self.support_score:
                continue
            if not (region[0] <= (b[0]+b[2])/2 < region[2] and region[1] <= (b[1]+b[3])/2 < region[3]):
                continue
            if any(iou(b, box) >= .5 and
                   min((box[i+2]-box[i])/(b[i+2]-b[i]) for i in (0, 1)) >= .6
                   for o in existing for box in self._identity_boxes(o)):
                continue
            rows.append(group)
        # Match known objects during object observation. Discovery uses its small
        # fixed budget for independent extents, not already matched measurements.
        rows.sort(key=lambda g: (-area(g['representative']['bbox']), -g['representative']['score'], g['id']))
        return rows[:self.region_limit], len(rows)

    @staticmethod
    def _identity_boxes(candidate):
        # Coverage knows actual output. A registered but empty fit does not
        # represent visible geometry; regional discovery only knows identity.
        if 'fitted_part_boxes' in candidate:
            return candidate['fitted_part_boxes']
        if 'fitted_bbox_2d' in candidate:
            return [candidate['fitted_bbox_2d']] if candidate['fitted_bbox_2d'] else []
        return [candidate['bbox_2d']]

    def review_validator(self, base, folder, existing, *, isolate_coverage_conflicts=False):
        path = Path(folder)/'sam_evidence.json'
        evidence = read(path)
        measurements = {r['id']: r for r in evidence['shown']}
        expected = set(measurements)
        if not expected:
            return base
        existing_ids = {o['id'] for o in existing}

        def source_box(box):
            frame = evidence.get('output_frame') or read(Path(folder)/'frame.json')
            crop, size = frame['source_crop_half_open'], frame['actual_crop_input_size_px']
            if frame.get('coordinate_mode') == 'norm1000':
                box = [v*(size[i % 2]-1)/1000 for i,v in enumerate(box)]
            return [(v+.5)*(crop[i % 2+2]-crop[i % 2])/size[i % 2]+crop[i % 2]-.5
                    for i,v in enumerate(box)]

        def validate(raw):
            raw = copy.deepcopy(raw)
            conflicts=raw.pop('sam_coverage_conflicts',[])
            rows = raw.pop('sam_decisions', None)
            if not isinstance(rows, list) or len(rows) != len(expected):
                raise ValueError('Return one sam_decisions entry for every displayed SAM ID')
            if any(not isinstance(row, dict) or not isinstance(row.get('sam_id'), str) for row in rows) or {r.get('sam_id') for r in rows} != expected:
                raise ValueError('sam_decisions must account for the exact displayed SAM IDs')
            result = base(raw)
            if conflicts:result['sam_coverage_conflicts']=conflicts
            boxes = {o['id']: self._identity_boxes(o) for o in existing}
            new_ids = {o['id']: i for i,o in enumerate(result['new_objects']) if 'id' in o}
            for review in result.get('reviews', []):
                if review['action'] == 'redraw':
                    boxes[review['id']] = [source_box(review['bbox_2d'])]
            for row in rows:
                if set(row) != {'sam_id', 'action', 'target', 'reason'} or not isinstance(row['reason'], str) or not row['reason'].strip():
                    raise ValueError('Each SAM disposition requires sam_id/action/target/reason')
                action, target = row['action'], row['target']
                if action == 'covered':
                    if isinstance(target, str) and target in existing_ids:
                        target_boxes = boxes[target]
                    else:
                        index = new_ids.get(target) if isinstance(target, str) else target
                        if type(index) is not int or not 0 <= index < len(result['new_objects']):
                            raise ValueError('covered SAM target must reference an existing or returned new object')
                        target_boxes = [source_box(result['new_objects'][index]['bbox_2d'])]
                    sam_box = measurements[row['sam_id']]['bbox_original']
                    # Use the SAME represented-extent test as discovery.
                    # Mere intersection lets a large wiring bbox certify a
                    # missing node in its empty interior.
                    if not any(iou(sam_box, box) >= .5 and
                               min((box[i+2]-box[i])/(sam_box[i+2]-sam_box[i]) for i in (0, 1)) >= .6
                               for box in target_boxes):
                        reason=(
                            f"{row['sam_id']}: covered contradicts the represented parts of {target}. "
                            'Recheck CLEAN and the part inventory; return a missing-part repair, '
                            'an independent discovery, or an explicit reject/uncertain disposition.')
                        if not isolate_coverage_conflicts:raise ValueError(reason)
                        # Reject this claim, not independent source discoveries
                        # in the same whole-image coverage reply. Never certify
                        # the disputed SAM fragment as represented.
                        result.setdefault('sam_coverage_conflicts',[]).append(copy.deepcopy(row))
                        row.update(action='uncertain',target=None,reason=reason)
                elif action == 'add':
                    if type(target) is not int or not 0 <= target < len(result['new_objects']):
                        raise ValueError('add SAM target must index a returned new_objects entry')
                elif action not in {'reject', 'uncertain'} or target is not None:
                    raise ValueError('reject/uncertain SAM targets must be null')
            result['sam_decisions'] = rows
            unresolved = [r['sam_id']+': '+r['reason'] for r in rows if r['action'] == 'uncertain']
            if unresolved:
                result['uncertain'] = list(dict.fromkeys([*result['uncertain'], *unresolved]))
                if 'status' in result:
                    result['status'] = 'uncertain'
                if 'more_pending' in result:
                    result['more_pending'] = True
            return result
        return validate

    def coverage(self, prompt, images, item, existing, folder, *, coordinate_mode='pixel'):
        """Carry unrepresented measurements into the existing full-image review."""
        crop = [0, 0, *item['source_size']]
        groups, count = self._unrepresented(crop, existing)
        target = Path(folder)/'sam_candidates.png'
        records = self.render(images[0][1], groups, crop, target, masks=False) if groups else []
        save(Path(folder)/'sam_evidence.json', {'stage': 'coverage', 'shown': records,
                                             'eligible': count, 'truncated': count > len(records),
                                             'output_frame': {'source_crop_half_open': crop,
                                                 'actual_crop_input_size_px': item['input_size'],
                                                 'coordinate_mode': coordinate_mode}})
        prompt = copy.deepcopy(prompt)
        # Coverage owns its rules. SAM adds evidence and never rewrites another
        # stage's prompt through a fragile exact-string substitution.
        boxes = {o['id']: local_box(o['bbox_2d'], crop, item['input_size']) for o in existing}
        for candidate in prompt['object_states']:
            candidate['identity_bbox_pixels'] = [round(v, 1) for v in boxes[candidate['id']]]
        fits = {o['id']: o['fitted_bbox_2d'] for o in existing if o.get('fitted_bbox_2d')}
        for candidate in prompt['object_states']:
            # A critic's unresolved verdict can still have valid fitted parts.
            # Coverage concerns visible presence, not that verdict.
            candidate.pop('state', None)
            candidate['has_fitted_geometry'] = candidate['id'] in fits
            candidate['fitted_bbox_pixels'] = ([round(v, 1) for v in local_box(fits[candidate['id']], crop, item['input_size'])]
                                               if candidate['id'] in fits else None)
        if not records:
            return prompt, images
        prompt['sam_auxiliary_evidence'] = {
            'candidates': [{'id': r['id'], 'category_hint': r['category'],
                            'bbox_pixels': [round(v, 1) for v in r['bbox_local']]} for r in records],
            'rule': 'SAM and identity boxes use pixels of the supplied full image and are uncertain search '
                    'hints. Resolve each against the clean source and existing inventory; output coordinates '
                    'still follow the declared output convention.'}
        prompt = self._accounting(prompt, records, formal=True)
        prompt['sam_candidate_accounting'] += (
            ' In this fitting coverage review, covered requires an actual represented_part '
            'matching the source feature; name that part in reason. A parent ID or its union '
            'bbox is not sufficient. Missing intrinsic content belongs in repair_existing; '
            'independent missing objects belong in new_objects.')
        return prompt, [*images, ('Unrepresented SAM proposals in the same full image', target)]

    def critic(self, prompt, images, job, folder, *, force=False):
        prompt, images = super().critic(prompt, images, job, folder, force=force)
        if 'bbox_local' in prompt.get('sam_auxiliary_evidence',{}).get('measurement',{}):
            fit = read(Path(folder)/'fit_local.json')['bbox_2d']
            measurement = prompt['sam_auxiliary_evidence']['measurement']
            size = job['frame']['actual_crop_input_size_px']
            if job.get('coordinate_mode') == 'norm1000':
                fit = [v*1000/(size[i % 2]-1) for i,v in enumerate(fit)]
            prompt['extent_comparison'] = {
                'coordinate_system': measurement['coordinate_system'],
                'fitted_bbox': [round(v, 2) for v in fit],
                'fit_minus_sam_left_top_right_bottom': [round(a-b, 2) for a,b in zip(fit, measurement['bbox_local'])],
                'rule': 'Use these differences to inspect the corresponding source sides and overall size. '
                        'They are not an error score: SAM measures foreground exterior, while stroked shapes '
                        'fit centerlines. Resolve mismatches from the clean source, not from box agreement.'}
            # A single matched primitive cannot silently dismiss a material
            # extent disagreement. This nominates a SOURCE check in the same
            # critic call; it never snaps geometry to the segmentation box.
            parts=prompt.get('fitted_parts',[])
            if (len(parts)==1 and parts[0].get('kind') in ('rectangle','rounded_rectangle','ellipse')
                    and measurement.get('mask_available') and measurement.get('target_extent_iou',0)>=.75):
                bounds=measurement['bbox_local']
                scale=[size[i]/job['frame']['source_crop_size'][i] for i in (0,1)]
                if job.get('coordinate_mode')=='norm1000':
                    scale=[scale[i]*1000/(size[i]-1) for i in (0,1)]
                flagged=[side for i,side in enumerate(('left','top','right','bottom'))
                         if abs(fit[i]-bounds[i])>max(4*scale[i%2],.02*(bounds[i%2+2]-bounds[i%2]))]
                if flagged:prompt['extent_comparison']['required_source_sides']=flagged
        return prompt, images

    def precritic(self, prompt, images, job, folder):
        prompt, images = self.critic(prompt, images, job, folder, force=True)
        if 'sam_auxiliary_evidence' in prompt:
            prompt['version'] = 'shape_integration.sam_precheck.1'
            prompt['checks'] = [
                'This preliminary check concerns only target extent and SAM-supported contour spans. '
                'Compare the measured extent with fitted extent, then inspect the corresponding sides '
                'in the clean source. Distinguish actual fill/stroke boundaries from crop boundaries, '
                'shadows and annotation edges. A difference is a search hint, not proof of an error.',
                'When a usable mask is present, inspect its reliable side and corner spans against '
                'the source and fitted paths. Do not infer a part inventory, hidden topology or missing '
                'children from the silhouette. The following original critic checks complete structure. '
                'Return only source-supported extent/contour findings; uncertain SAM measurements '
                'must not force a geometry correction.']
            prompt.pop('shared_vertices', None)
        return prompt, images
