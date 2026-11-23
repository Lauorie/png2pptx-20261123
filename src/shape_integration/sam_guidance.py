"""One matched SAM support: observation extent, boundary evidence, optional critic.

All geometry remains produced by the existing points/parts fitter. Detection
extent and segmentation boundaries are fallible measurements, not target truth.
"""
import copy
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import save
from .sam_evidence import SamEvidence, iou, local_box, project_mask


@lru_cache(maxsize=2048)
def mask_continuity(path):
    """Disconnected foreground supports visual evidence, not one contour's samples."""
    with Image.open(path) as im:
        mask = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'empty'
    xx, yy, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    y, x = np.nonzero(mask)
    coherent = min(width / (x.max()-x.min()+1), height / (y.max()-y.min()+1)) >= .7
    return 'continuous' if coherent else 'fragmented'


class GuidedSamEvidence(SamEvidence):
    def __init__(self, groups, assets=(), *, mode='mask', support_score=.7,
                 support_iou=.2, observer_numeric_bbox=True, mask_iou=0., activation='all', **kwargs):
        super().__init__(groups, assets, **kwargs)
        if mode not in {'bbox', 'mask', 'critic'}:
            raise ValueError('Unknown SAM guidance mode')
        self.mode = mode
        if activation not in {'all', 'repair'}:
            raise ValueError('Unknown SAM activation policy')
        self.activation = activation
        self.support_score, self.support_iou = support_score, support_iou
        self.observer_numeric_bbox, self.mask_iou = observer_numeric_bbox, mask_iou

    def region(self, prompt, images, frame, assignment, folder):
        if self.activation == 'repair':
            return prompt, images
        return super().region(prompt, images, frame, assignment, folder)

    def support(self, candidate, *, min_score=None):
        """Allow imperfect localization; reject a proposal claimed more strongly by a peer."""
        rows = []
        for group in self.groups:
            p = group['representative']
            match = iou(p['bbox'], candidate['bbox_2d'])
            threshold=self.support_score if min_score is None else min_score
            # A precise local extent can be more informative than a high-score
            # enclosing strip. Lower-score proposals remain auxiliary and need
            # strong target overlap; they never supply fitted coordinates.
            precise_auxiliary=(min_score is None and match>=.75 and p['score']>=max(.45,self.support_score-.25))
            if not self.eligible(group) or (p['score'] < threshold and not precise_auxiliary) or match < self.support_iou:
                continue
            box = candidate['bbox_2d']
            if min((p['bbox'][i+2]-p['bbox'][i]) / (box[i+2]-box[i]) for i in (0, 1)) < .6:
                continue  # A small constituent cannot specify its parent's total size.
            peers = [iou(p['bbox'], c['bbox_2d']) for c in self.inventory if c['id'] != candidate['id']]
            if peers and max(peers) > match + .05:
                continue
            rows.append((.8 * match + .2 * p['score'], group))
        return max(rows, key=lambda x: (x[0], x[1]['id']))[1] if rows else None

    def geometry_support(self, candidate):
        """Low-score cached masks need source verification, not score-only trust.

        Keep target/peer ownership and scale checks. The numeric guard must
        independently verify distributed source agreement before any veto.
        This does not expose more proposals to GLM or alter object discovery.
        """
        return self.support(candidate, min_score=0.)

    def prepare(self, candidate, crop, source_size):
        group = self.support(candidate)
        if group is None:
            return crop, None
        p = group['representative']
        b = p['bbox']
        context_box=b
        mask_box=p.get('mask_bbox')
        if (mask_box and self.mask_usable(group,candidate) and p.get('mask_bbox_iou',0)>=.5):
            # Detection boxes can cut off a real stepped contour even when
            # their mask contains it. Extend context, never target identity.
            context_box=[min(b[0],mask_box[0]),min(b[1],mask_box[1]),
                         max(b[2],mask_box[2]),max(b[3],mask_box[3])]
        if self.activation == 'repair' and (crop[0] <= context_box[0] and crop[1] <= context_box[1]
                                           and context_box[2] < crop[2] and context_box[3] < crop[3]):
            return crop, None  # Existing observation already contains the support.
        pad = .12 * max(context_box[2] - context_box[0], context_box[3] - context_box[1])
        support_crop = [max(0, math.floor(context_box[0] - pad)), max(0, math.floor(context_box[1] - pad)),
                        min(source_size[0], math.ceil(context_box[2] + pad) + 1),
                        min(source_size[1], math.ceil(context_box[3] + pad) + 1)]
        expanded = [min(crop[0], support_crop[0]), min(crop[1], support_crop[1]),
                    max(crop[2], support_crop[2]), max(crop[3], support_crop[3])]
        # A loosely matched enclosing mask may be useful context, but must not
        # silently rename a nested target to the surrounding frame.
        agrees = iou(b, candidate['bbox_2d']) >= .5
        if not agrees:
            expanded = list(crop)
        return expanded, {'id': group['id'], 'bbox_original': b,
                          'identity_bbox_original': candidate['bbox_2d'],
                          'identity_source': 'candidate' if agrees else 'candidate_sam_extent_disagrees',
                          'category': p['category'],
                          'context_bbox_original':context_box,
                          'candidate_score': p['score'], 'original_crop': crop,
                          'expanded_crop': expanded, 'geometry_overwrite': False,
                          'context_expansion_withheld': not agrees}

    def mask_usable(self, group, candidate=None):
        p = group['representative']
        frame, box = p['frame']['box'], p['bbox']
        pixels = self.processor_resolution * min((box[2] - box[0]) / (frame[2] - frame[0]),
                                                 (box[3] - box[1]) / (frame[3] - frame[1]))
        agrees = candidate is None or iou(box, candidate['bbox_2d']) >= self.mask_iou
        return bool(p.get('mask_quality_ok') and pixels >= self.min_mask_short_side and agrees
                    and mask_continuity(str(p['mask_path'])) != 'empty')

    def boundary_samples(self, group, crop, size):
        if mask_continuity(str(group['representative']['mask_path'])) != 'continuous':
            return []  # A single dash must not stand in for the whole object's contour.
        mask = np.asarray(project_mask(group['representative'], crop, size)) > 0
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        boundary = max(contours, key=cv2.contourArea).reshape(-1, 2)
        # Bounded measurements preserve corner locations better than a dense
        # overlay alone. These samples never enter the fitter as observations.
        perimeter = cv2.arcLength(boundary.reshape(-1, 1, 2), True)
        points = cv2.approxPolyDP(boundary.reshape(-1, 1, 2), max(1., .002 * perimeter), True).reshape(-1, 2)
        if len(points) > 32:
            points = points[np.linspace(0, len(points) - 1, 32).astype(int)]
        return points.tolist()

    def _evidence(self, images, job, folder, *, critic=False):
        if self.activation == 'repair' and not (job.get('sam_support') or job.get('revision_feedback')):
            return None, images
        group = self.support(job['candidate'])
        if group is None:
            return None, images
        if iou(group['representative']['bbox'],job['candidate']['bbox_2d']) < .5:
            return None, images  # Do not prime a local fit with its enclosing mask.
        crop = job['frame']['source_crop_half_open']
        size = job['frame']['actual_crop_input_size_px']
        mask = self.mode != 'bbox' and self.mask_usable(group, job['candidate'])
        if (not critic and not self.observer_numeric_bbox and not mask
                and not job.get('sam_support')):
            return None, images  # No new crop, numeric extent or usable boundary to show.
        p = group['representative']
        b = local_box(p['bbox'], crop, size)
        measurement = {'id': group['id'], 'bbox_local': [round(v, 1) for v in b],
                       'size_local': [round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                       'mask_available': mask,
                       'detector_category_hint':p['category'],
                       'target_extent_iou':round(iou(p['bbox'],job['candidate']['bbox_2d']),3),
                       'detector_score':p['score'],
                       'category_authority':'fallible_location_and_shape_family_hint_only'}
        if mask and p.get('mask_bbox') and .5 <= p.get('mask_bbox_iou',1.) < .85:
            measurement['detector_mask_extent_conflict'] = (
                'The detector rectangle and its segmented foreground have different extents. '
                'The identity box may truncate a connected contour. Trace the complete target in CLEAN; '
                'neither the detector box nor a merged neighboring mask defines its shape. '
                'Check every apparent closing side against actual source ink, including stepped '
                'continuations. Keep genuinely distinct nested contours separate.')
        output_images = list(images)
        if mask:
            path = Path(folder) / ('sam_critic.png' if critic else 'sam_contours.png')
            measurement['mask_continuity'] = mask_continuity(str(p['mask_path']))
            if measurement['mask_continuity'] == 'fragmented':
                # The source is already Image 1. Isolate disconnected support
                # so unsegmented scene strokes cannot look like mask evidence.
                view = Image.new('RGB', tuple(size), 'white')
                view.paste('#b000b5', (0, 0, *size), project_mask(p, crop, size))
                view.save(path)
                measurement['mask_view'] = 'isolated_foreground_fragments'
            else:
                self.render(Path(folder) / 'clean.png', [group], crop, path, masks=True)
            # The backend uses cached mask pixels for geometric validation.
            # Avoid asking GLM to reinterpret a second numeric contour point
            # table in addition to the aligned mask image and original crop.
            output_images.append(('SAM mask boundary in the SAME crop; uncertain auxiliary measurement', path))
        measurement['coordinate_system'] = job.get('coordinate_mode', 'pixel')
        if job.get('coordinate_mode') == 'norm1000':
            factors = [1000 / (v - 1) for v in size]
            measurement['bbox_local'] = [round(v * factors[i % 2], 2) for i, v in enumerate(b)]
            measurement['size_local'] = [round((b[i + 2] - b[i]) * factors[i], 2) for i in (0, 1)]
            if 'mask_boundary_samples_local' in measurement:
                measurement['mask_boundary_samples_local'] = [[round(v * factors[i], 2) for i, v in enumerate(pt)]
                                                               for pt in measurement['mask_boundary_samples_local']]
        save(Path(folder) / ('sam_critic_evidence.json' if critic else 'sam_evidence.json'),
             {'stage': 'critic' if critic else 'object_observation', 'mode': self.mode,
              'object_id': job['object_id'], 'shown': [measurement],
              'activation': self.activation,
              'activation_reason': ('crop_extension' if job.get('sam_support') else 'existing_revision')
                                   if self.activation == 'repair' else 'all_matched',
              'geometry_overwrite': False, 'source_bbox': p['bbox'],
              'observer_numeric_bbox': self.observer_numeric_bbox, 'mask_iou': self.mask_iou})
        if not self.observer_numeric_bbox and not critic:
            measurement = {k: v for k, v in measurement.items() if k not in {'bbox_local', 'size_local'}}
            measurement['extent_reference'] = (
                'Image 2 retains the original target identity box; available SAM mask evidence is auxiliary.')
        return measurement, output_images

    def observe(self, prompt, images, job, folder):
        measurement, images = self._evidence(images, job, folder)
        if measurement is None:
            save(Path(folder) / 'sam_evidence.json', {'stage': 'object_observation', 'mode': self.mode,
                                                   'object_id': job['object_id'], 'shown': []})
            return prompt, images
        prompt = copy.deepcopy(prompt)
        prompt['sam_auxiliary_evidence'] = {
            'measurement': measurement,
            'rules': [
                'The measured box is an OUTER foreground extent and can include stroke thickness or shadows. '
                'Use it to locate the target and estimate its full width and height, not as fitted '
                'stroke-centerline bounds or four shape corners. Resolve geometry from clean Image 1.',
                'When available, mask samples describe the segmented outer boundary in the stated coordinates. '
                'Use reliable spans to locate sides and curved-to-straight transitions, then place observations '
                'on the clean source stroke CENTERLINE (fill boundary when unstroked), excluding shadows.',
                'A fragmented mask shows disconnected foreground strokes, such as dashes, as VISUAL evidence only. '
                'Use supported stroke locations; gaps do not mean missing shape parts. Do not connect mask fragments '
                'into an invented contour or treat one fragment as the entire target boundary.',
                'A single silhouette never defines the part list. Respect the current initial-observation or '
                'part-edit scope. SAM measurements cannot authorize extra parts or changes to preserved parts. '
                'Preserve the distinction between a container and its children.',
                'For rounded rectangles use actual arc-interior points between the tangent transitions; '
                'straight-side points cannot determine corner curvature. Ignore an inaccurate mask span.',
                'If target geometry is cut by the crop, request context; do not clamp unseen corners or close a '
                'shape at the crop boundary. Keep the original points/parts schema and task scope.',
                'The SAM category is a fallible coarse shape-family hint, not an output type. '
                'Confirm corners, curvature, gaps, pointers and connected boundaries in CLEAN. '
                'A rectangle detection may describe only the body of a callout or a parent panel. '
                'Never replace visible notches, pointers or curves with the detector template.'
            ]}
        if measurement.get('mask_continuity') == 'fragmented':
            prompt['sam_auxiliary_evidence']['rules'].append(
                'The SAM image isolates foreground fragments on white in the SAME crop coordinates. '
                'Other scene strokes are deliberately absent from that view, not reported as missing. '
                'Distinguish a continuous target contour from lines branching away from it in the clean '
                'source. An attached line does not replace an intact side or authorize a tail/notch.')
        if not self.observer_numeric_bbox:
            prompt['sam_auxiliary_evidence']['rules'][0] = (
                'Image 2 retains the original target identity box; use any SAM contour only as an auxiliary cue. '
                'Observe the actual contour in clean Image 1; do not copy annotation edges as geometry.')
        if job.get('sam_support', {}).get('identity_source') == 'candidate_sam_extent_disagrees':
            prompt['sam_auxiliary_evidence']['rules'].append(
                'The SAM extent disagrees with the localized target. Image 2 keeps the original target '
                'identity box; the SAM image may include an enclosing or neighboring object. Do not '
                'substitute that larger contour for this target. Resolve identity from clean Image 1.')
        return prompt, images

    def critic(self, prompt, images, job, folder, *, force=False):
        if self.mode != 'critic' and not force:
            return prompt, images
        measurement, images = self._evidence(images, job, folder, critic=True)
        if measurement is None:
            return prompt, images
        prompt = copy.deepcopy(prompt)
        prompt['sam_auxiliary_evidence'] = {
            'measurement': measurement,
            'rule': 'Use this uncertain extent/boundary measurement to locate possible size or curvature errors. '
                    'The box and mask describe OUTER foreground extent, which can include stroke thickness '
                    'or shadows; fitted stroke-centerline bounds must be judged from the clean source. '
                    'Every acceptance or correction must still be supported by the CLEAN SOURCE. '
                    'A fragmented mask shows separate foreground strokes as visual evidence only; '
                    'check their supported locations without inventing connections or interpreting gaps as missing parts. '
                    'Do not judge mask agreement as correctness or treat the mask as an exhaustive part list.'}
        if measurement.get('detector_mask_extent_conflict'):
            prompt['sam_detector_mask_extent_conflict']=measurement['detector_mask_extent_conflict']
        if measurement.get('mask_continuity') == 'fragmented':
            prompt['sam_auxiliary_evidence']['rule'] += (
                ' The SAM image isolates foreground fragments on white in the SAME crop coordinates. '
                'Do not mistake other lines visible only in the clean source for segmented support. '
                'Before replacing a side by a tail/notch, verify that the target contour itself is '
                'interrupted or replaced there; an intact side with a branching leader stays separate.')
        return prompt, images
