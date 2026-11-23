"""Optional SAM observations for existing reviews; never modify shape geometry.

The producer supplies original-image bbox coordinates and a mask crop frame.
The same provider is shared by initial observations and bounded revisions.
No network calls, per-object inference, or extra critic calls occur here.
"""
import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from .common import save


def area(b):
    return max(0., b[2]-b[0])*max(0., b[3]-b[1])


def overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    intersection = overlap(a, b)
    return intersection/max(1e-9, area(a)+area(b)-intersection)


def local_box(box, crop, size):
    return [(v-crop[i % 2]+.5)*size[i % 2]/(crop[i % 2+2]-crop[i % 2])-.5
            for i, v in enumerate(box)]


def project_mask(prediction, crop, size):
    """Pillow edge-coordinate affine mapping preserves pixel-center transport."""
    with Image.open(prediction['mask_path']) as im:
        mask = im.convert('L')
        box = prediction['frame']['box']
        sx, sy = (box[2]-box[0])/mask.width, (box[3]-box[1])/mask.height
        affine = ((crop[2]-crop[0])/size[0]/sx, 0., (crop[0]-box[0])/sx,
                  0., (crop[3]-crop[1])/size[1]/sy, (crop[1]-box[1])/sy)
        return mask.transform(tuple(size), Image.Transform.AFFINE, affine,
                              resample=Image.Resampling.NEAREST, fillcolor=0)


class SamEvidence:
    def __init__(self, groups, assets=(), *, region_limit=32, object_limit=3, object_iou=.45,
                 min_mask_short_side=64, processor_resolution=1008):
        self.groups = groups
        self.assets = assets
        self.region_limit, self.object_limit, self.object_iou = region_limit, object_limit, object_iou
        self.min_mask_short_side, self.processor_resolution = min_mask_short_side, processor_resolution
        self.inventory = []

    def set_inventory(self, inventory):
        self.inventory = copy.deepcopy(inventory)

    def eligible(self, group):
        box = group['representative']['bbox']
        # An image asset owns its interior. This does not reject an enclosing container.
        return not any(overlap(box, a['bbox_2d'])/max(1e-9, area(box)) > .9
                       and area(box) <= area(a['bbox_2d'])*1.1 for a in self.assets)

    def select_region(self, crop, candidates):
        rows = [g for g in self.groups if self.eligible(g)
                and crop[0] <= sum(g['representative']['bbox'][::2])/2 < crop[2]
                and crop[1] <= sum(g['representative']['bbox'][1::2])/2 < crop[3]]
        # Unrepresented candidates first; matching candidates still help localization.
        rows.sort(key=lambda g: (max((iou(g['representative']['bbox'], c['bbox_2d'])
                                     for c in candidates), default=0) >= .5,
                                 -g['representative']['score'], g['id']))
        return rows[:self.region_limit], len(rows)

    def select_object(self, candidate):
        rows = []
        for group in self.groups:
            p = group['representative']; match = iou(p['bbox'], candidate['bbox_2d'])
            if not self.eligible(group) or match < self.object_iou:
                continue
            # SAM3 stretches each input to a square. Measure actual support in
            # that network frame, rather than pixels in an upscaled PNG mask.
            frame=p['frame']['box'];box=p['bbox']
            support=self.processor_resolution*min((box[2]-box[0])/(frame[2]-frame[0]),
                                                   (box[3]-box[1])/(frame[3]-frame[1]))
            if not p.get('mask_quality_ok',False) or support<self.min_mask_short_side:
                continue  # Its bbox was already available in regional review.
            other = max((iou(p['bbox'], c['bbox_2d']) for c in self.inventory
                         if c['id'] != candidate['id']), default=0.)
            if other > match+.15:
                continue
            rows.append((match, group))
        rows.sort(key=lambda pair: (-pair[0], -pair[1]['representative']['score'], pair[1]['id']))
        return [g for _, g in rows[:self.object_limit]]

    def render(self, clean_path, groups, crop, path, *, masks):
        with Image.open(clean_path) as im:
            image = im.convert('RGB'); size = image.size
        draw = ImageDraw.Draw(image)
        records = []
        for group in groups:
            p = group['representative']; b = local_box(p['bbox'], crop, size)
            usable = masks and p.get('mask_quality_ok', False)
            if usable:
                mask = np.asarray(project_mask(p, crop, size)) > 0
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    points = [tuple(map(int, point)) for point in contour.reshape(-1, 2)]
                    if len(points) > 2:
                        draw.line(points+[points[0]], fill='#b000b5', width=2)
            if not masks or not usable:
                draw.rectangle(b, outline='#c77b00', width=2)
            x, y = max(0, min(size[0]-70, b[0])), max(0, min(size[1]-15, b[1]-14))
            draw.text((x, y), group['id'], fill='#b000b5', stroke_width=1, stroke_fill='white')
            records.append({'id': group['id'], 'category': p['category'], 'bbox_original': p['bbox'],
                            'bbox_local': b, 'candidate_score': p['score'], 'mask_shown': bool(usable),
                            'mask_path': p['mask_path'], 'frame': p['frame']})
        image.save(path)
        return records

    def region(self, prompt, images, frame, assignment, folder):
        crop = frame['source_crop_half_open']
        groups, total = self.select_region(crop, assignment['candidates'])
        path = Path(folder)/'sam_candidates.png'
        records = self.render(images[0][1], groups, crop, path, masks=False) if groups else []
        save(Path(folder)/'sam_evidence.json', {'stage': 'regional_review', 'eligible': total,
              'shown': records, 'truncated': total > len(groups), 'not_exhaustive': True})
        if not groups:
            return prompt, images
        prompt = copy.deepcopy(prompt)
        prompt['sam_auxiliary_evidence'] = {
            'candidates': [{'id': r['id'], 'category_hint': r['category']} for r in records],
            'rules': ['Image 3 shows fallible SAM localization proposals, not additional confirmed objects.',
                      'Use clean Image 1 to resolve identity, missed objects and incomplete boxes. SAM absence never justifies rejecting a GLM object.',
                      'Match against ALL assigned and context objects. Preserve compound objects and distinct container/child identities; do not duplicate a parent with its components.',
                      'SAM may miss shapes, merge neighbors or detect image content. Search the entire work area even beyond shown candidates. Confidence is not contour accuracy.',
                      'Keep the original output schema and IDs; new objects require direct clean-image evidence.']}
        return prompt, [*images, ('Image 3: aligned SAM candidate boxes; annotations are not source shapes', path)]

    def observe(self, prompt, images, job, folder):
        groups = self.select_object(job['candidate'])
        path = Path(folder)/'sam_contours.png'
        records = self.render(images[0][1], groups, job['frame']['source_crop_half_open'], path, masks=True) if groups else []
        save(Path(folder)/'sam_evidence.json', {'stage': 'object_observation', 'object_id': job['object_id'],
              'shown': records, 'not_exhaustive': True, 'geometry_overwrite': False,
              'min_mask_short_side_network_px':self.min_mask_short_side})
        if not groups:
            return prompt, images
        prompt = copy.deepcopy(prompt)
        prompt['sam_auxiliary_evidence'] = {
            'rules': ['Image 3 is aligned to Images 1 and 2. Magenta curves are uncertain mask boundaries; orange boxes are localization hints only.',
                      'Observe ALL parts of the identified object using clean Image 1. These proposals are incomplete evidence, never an exhaustive part list.',
                      'For stroked shapes place observations on the stroke CENTERLINE, not mask exterior, shadow or annotation. A mask can enlarge an object by a stroke width.',
                      'Resolve straight-to-curved transitions from the source. For rounded rectangles retain the existing 16-point observation protocol; do not assume a pill or copy a mask radius.',
                      'Preserve source-supported equal sizes, gaps, alignment, face junctions and shared point IDs. Ignore a conflicting mask. Return the unchanged points/parts schema.']}
        return prompt, [*images, ('Image 3: SAM contour evidence in the SAME local coordinates; clean image authoritative', path)]
