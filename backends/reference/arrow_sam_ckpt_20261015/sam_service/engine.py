"""Single-GPU SAM3 adapter using the existing training snapshot and weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def select_categories(requested, available):
    if requested == 'all':
        return list(available)
    if not isinstance(requested, list) or not requested:
        raise ValueError('categories must be a nonempty list of manifest names or "all"')
    if any(not isinstance(name, str) for name in requested):
        raise ValueError('Each category must be a string')
    by_name = {category['name']: category for category in available}
    unknown = set(requested) - by_name.keys()
    if unknown:
        raise ValueError('Unknown categories: ' + ', '.join(sorted(unknown)))
    return [by_name[name] for name in dict.fromkeys(requested)]


def mask_geometry(mask):
    """Preserve all components and holes; keep legacy largest contour as a hint."""
    import cv2
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []
    entries = [{'points': contour.reshape(-1, 2).tolist(),
                'parent': int(hierarchy[0][index][3])}
               for index, contour in enumerate(contours)]
    largest = max(range(len(contours)), key=lambda i: abs(cv2.contourArea(contours[i])))
    return entries[largest]['points'], entries


def request_threshold(request, default=0.5):
    value = request.get('threshold', default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('threshold must be a finite number in [0, 1]')
    return float(value)


def save_probability_crop(probabilities, output, ident, floor):
    """Lossless float32 crop covering every pixel above the supported floor."""
    import numpy as np
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError('Expected a finite two-dimensional probability map in [0, 1]')
    rows, cols = np.nonzero(values > floor)
    if len(rows):
        x, y, right, bottom = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
        crop = values[y:bottom, x:right]
    else:
        x = y = right = bottom = 0
        crop = np.empty((0, 0), dtype=np.float32)
    path = Path(output) / (ident + '.probabilities.npz')
    np.savez_compressed(path, probabilities=crop)
    return {'probability_path': str(path), 'probability_rect': [x, y, right - x, bottom - y],
            'probability_floor': floor, 'probability_dtype': 'float32'}


class SamEngine:
    def __init__(self, config):
        import torch
        training = Path(config['training_run']).resolve()
        repository = Path(config.get('sam_repository',training / 'runtime/third_party/sam3'))
        sys.path.insert(0, str(repository))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one allocated GPU to this service')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This SAM3 configuration requires BF16 support')
        self.torch = torch
        self.config = config
        manifest = Path(config['dataset_manifest'])
        self.available = json.loads(manifest.read_text())['categories']
        self.defaults = select_categories(config['categories'], self.available)
        checkpoint_path = Path(config['checkpoint_path'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        self.checkpoint_info = {'epoch': int(checkpoint['epoch']),
            'model_parameter_keys': len(checkpoint['model']),
            'selection': 'user-requested ' + config['checkpoint_name'] + ' frozen before inference',
            'sha256': config['checkpoint_sha256']}
        if 'expected_checkpoint_epoch' in config and int(checkpoint['epoch']) != config['expected_checkpoint_epoch']:
            raise ValueError('Frozen checkpoint epoch does not match best.json')
        model = build_sam3_image_model(
            bpe_path=str(repository / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
            device='cpu', load_from_HF=False, enable_segmentation=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        self.model = model.cuda().eval()
        self.processor = Sam3Processor(self.model, resolution=int(config['resolution']),
                                       confidence_threshold=float(config['sam_threshold']))
        properties = torch.cuda.get_device_properties(0)
        gpu_uuid = str(properties.uuid)
        if not gpu_uuid.startswith('GPU-'):
            gpu_uuid = 'GPU-' + gpu_uuid
        self.metadata = {'checkpoint_path': str(checkpoint_path), 'checkpoint': self.checkpoint_info,
                         'checkpoint_bytes': checkpoint_path.stat().st_size,
                         'checkpoint_mtime_ns': checkpoint_path.stat().st_mtime_ns,
                         'sam_repository': str(repository), 'gpu_name': properties.name,
                         'gpu_uuid': gpu_uuid, 'torch_version': torch.__version__,
                         'categories': self.available, 'default_categories': self.defaults,
                         'resolution': int(config['resolution']),
                         'sam_threshold': float(config['sam_threshold']),
                         'per_request_threshold': True,
                         'probability_export': True,
                         'score_comparison': 'strictly_greater', 'mask_pixel_threshold': 0.5}

    def warmup(self):
        from PIL import Image
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            state = self.processor.set_image(Image.new('RGB', (128, 128), 'white'))
            self.processor.set_text_prompt(self.defaults[0]['name'], state)
        self.torch.cuda.synchronize()

    def validate(self, request):
        if not isinstance(request, dict):
            raise ValueError('Request must be a JSON object')
        request_threshold(request, self.config.get('sam_threshold', 0.5))
        if not isinstance(request.get('save_probabilities', False), bool):
            raise ValueError('save_probabilities must be a boolean')
        floor = request_threshold({'threshold': request.get('probability_floor', 0.01)})
        if floor > 0.5:
            raise ValueError('probability_floor must be in [0, 0.5]')
        if not isinstance(request.get('image'), str) or not Path(request['image']).is_absolute():
            raise ValueError('image must be an absolute file path visible to the worker')
        if not Path(request['image']).is_file():
            raise ValueError('Input image does not exist on the worker')
        select_categories(request.get('categories', self.config['categories']), self.available)
        if 'output' in request and (not isinstance(request['output'], str)
                                   or not Path(request['output']).is_absolute()):
            raise ValueError('output must be an absolute directory path')

    def __call__(self, request):
        import numpy as np
        from PIL import Image, ImageDraw
        self.validate(request)
        threshold = request_threshold(request, self.config['sam_threshold'])
        export_probabilities = request.get('save_probabilities', False)
        probability_floor = request.get('probability_floor', 0.01)
        started = time.monotonic()
        image_path = Path(request['image'])
        image = Image.open(image_path).convert('RGB')
        output = Path(request['output'])
        # A request owns a fresh directory. No overwriting previous experiments.
        output.mkdir(parents=True, exist_ok=False)
        categories = select_categories(request.get('categories', self.config['categories']), self.available)
        save(output / 'request.json', {**request, 'threshold': threshold,
             'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()})
        predictions = []
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            # Every request sets its own value, including requests using the default.
            # The server serializes inference, so this processor is never shared concurrently.
            self.processor.set_confidence_threshold(threshold)
            state = self.processor.set_image(image)
            for category in categories:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(category['name'], state)
                for index, (box, score, tensor) in enumerate(zip(state['boxes'], state['scores'], state['masks'])):
                    mask = tensor.squeeze().cpu().numpy().astype(np.uint8)
                    if not mask.any() and not export_probabilities:
                        continue
                    contour, contours = mask_geometry(mask)
                    ident = f'S{len(predictions) + 1:03d}'
                    mask_path = output / f'{ident}.png'
                    Image.fromarray(mask * 255).save(mask_path)
                    predictions.append({'id': ident, 'category': category['name'],
                        'category_id': category['id'], 'score': float(score),
                        'bbox': box.float().cpu().tolist(), 'contour': contour,
                        'contours': contours, 'mask_path': str(mask_path), 'area': int(mask.sum())})
                    if export_probabilities:
                        # Despite its name, this processor's masks_logits is already sigmoid probability.
                        probabilities = state['masks_logits'][index].squeeze().float().cpu().numpy()
                        predictions[-1].update(save_probability_crop(probabilities, output, ident, probability_floor))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for pred in predictions:
            for contour in pred['contours']:
                points = [tuple(point) for point in contour['points']]
                if len(points) > 1:
                    draw.line(points + points[:1], fill='#d000c0', width=2)
            draw.text(tuple(pred['bbox'][:2]), pred['id'] + ' ' + pred['category'], fill='#d000c0')
        overlay.save(output / 'overlay.png')
        result = {'predictions': predictions, 'image_size': list(image.size),
                  'threshold': threshold, 'score_comparison': 'strictly_greater',
                  'probability_exported': export_probabilities,
                  'probability_floor': probability_floor if export_probabilities else None,
                  'checkpoint': self.checkpoint_info, 'model': self.metadata,
                  'seconds': time.monotonic() - started, 'output': str(output),
                  'overlay': str(output / 'overlay.png')}
        save(output / 'predictions.json', result)
        return result
