"""Persistent trained-SAM service, reachable on the pod network for host-side tile requests."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
import traceback


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class SamEngine:
    def __init__(self, config):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.threshold = config['sam_threshold']
        training = Path(config['training_run'])
        checkpoint_path = Path(config.get('sam_checkpoint', training / 'checkpoints/best.pt'))
        self.best = json.loads(checkpoint_path.with_suffix('.json').read_text())
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                mmap=True, weights_only=False)
        if int(checkpoint['epoch']) != self.best['epoch']:
            raise ValueError('best.pt epoch does not match best.json')
        bpe = training / 'runtime/third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
        model = build_sam3_image_model(bpe_path=str(bpe), device='cpu',
                                      load_from_HF=False, enable_segmentation=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        self.model = model.cuda().eval()
        self.processor = Sam3Processor(self.model, resolution=1008,
                                       confidence_threshold=config['sam_threshold'])
        manifest = json.loads((Path(config['dataset_root']) / 'dataset_manifest.json').read_text())
        self.categories = [c for c in manifest['categories'] if c['name'] not in {'line', 'block_arrow'}]

    def __call__(self, request):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
        image = Image.open(request['image']).convert('RGB')
        output = Path(request['output'])
        output.mkdir(parents=True, exist_ok=True)
        predictions = []
        started = time.monotonic()
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            state = self.processor.set_image(image)
            for category in (request.get('prompts') or self.categories):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(category['name'], state)
                if category.get('box') is not None:
                    state = self.processor.add_geometric_prompt(box=category['box'], label=True, state=state)
                for box, score, tensor in zip(state['boxes'], state['scores'], state['masks']):
                    mask = tensor.squeeze().cpu().numpy().astype(np.uint8)
                    if not mask.any():
                        continue
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(contour, True)
                    polygon = cv2.approxPolyDP(contour, max(0.7, perimeter * 0.0015), True).reshape(-1, 2)
                    if len(polygon) > 128:
                        polygon = polygon[np.linspace(0, len(polygon)-1, 128).astype(int)]
                    ident = f'S{len(predictions)+1:03d}'
                    mask_path = output / f'{ident}.png'
                    Image.fromarray(mask * 255).save(mask_path)
                    predictions.append({'id': ident, 'category': category['name'],
                        'category_id': category.get('id', -1), 'score': float(score),
                        'bbox': box.float().cpu().tolist(), 'contour': polygon.tolist(),
                        'mask_path': str(mask_path), 'area': int(mask.sum())})
        if not request.get('overlay', True):
            result = {'predictions': predictions, 'image_size': list(image.size),
                      'candidate_threshold': self.threshold, 'best_checkpoint': self.best,
                      'seconds': time.monotonic()-started, 'overlay': None}
            save(output / 'predictions.json', result)
            return result
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for pred in predictions:
            if len(pred['contour']) > 1:
                points = [tuple(p) for p in pred['contour']]
                draw.line(points + points[:1], fill='#d000c0', width=2)
            x, y = pred['bbox'][:2]
            draw.text((x, y), pred['id'] + ' ' + pred['category'], fill='#d000c0')
        overlay.save(output / 'overlay.png')
        result = {'predictions': predictions, 'image_size': list(image.size),
                  'candidate_threshold': self.threshold, 'best_checkpoint': self.best, 'seconds': time.monotonic()-started,
                  'overlay': str(output / 'overlay.png')}
        save(output / 'predictions.json', result)
        return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', choices=['sam'])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    save(args.state, {'phase': 'loading', 'engine': args.engine})
    engine = SamEngine(config)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ready"}')

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = engine(request)
                status = 200
            except Exception:
                result = {'error': traceback.format_exc()}
                print(result['error'], flush=True)
                status = 500
            payload = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(('0.0.0.0', args.port), Handler)
    save(args.state, {'phase': 'ready', 'engine': args.engine, 'port': args.port})
    server.serve_forever()


if __name__ == '__main__':
    main()
