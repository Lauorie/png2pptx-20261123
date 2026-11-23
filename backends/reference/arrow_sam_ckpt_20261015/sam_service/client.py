#!/usr/bin/env python3
"""Stdlib-only client. Image and output paths refer to the worker filesystem."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


class SamClient:
    def __init__(self, url, timeout=900):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self):
        with self.opener.open(self.url + '/health', timeout=5) as response:
            return json.load(response)

    def segment(self, image, output=None, categories=None, threshold=None, save_probabilities=False, probability_floor=0.01):
        body = {'image': str(Path(image).resolve())}
        if output is not None:
            body['output'] = str(Path(output).resolve())
        if categories is not None:
            body['categories'] = categories
        if threshold is not None:
            body['threshold'] = threshold
        if save_probabilities:
            body.update(save_probabilities=True, probability_floor=probability_floor)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('SAM request exceeded its timeout, including queue wait')
            request = urllib.request.Request(self.url + '/v1/segment', json.dumps(body).encode(),
                                             {'Content-Type': 'application/json'})
            try:
                with self.opener.open(request, timeout=remaining) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode()
                if error.code != 503:
                    raise RuntimeError(f'SAM HTTP {error.code}: {detail}') from error
                time.sleep(min(1, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument('--url')
    endpoint.add_argument('--run-dir', type=Path)
    parser.add_argument('--health', action='store_true')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--categories', nargs='+', help='Manifest names, or all')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--threshold', type=float, help='Per-request confidence threshold in [0, 1]')
    parser.add_argument('--save-probabilities', action='store_true')
    parser.add_argument('--probability-floor', type=float, default=0.01)
    args = parser.parse_args()
    url = args.url or json.loads((args.run_dir / 'endpoint.json').read_text())['url']
    client = SamClient(url, args.timeout)
    if args.health:
        result = client.health()
    else:
        if args.image is None:
            parser.error('--image is required unless --health is used')
        categories = 'all' if args.categories == ['all'] else args.categories
        result = client.segment(args.image, args.output, categories, threshold=args.threshold,
                                save_probabilities=args.save_probabilities, probability_floor=args.probability_floor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
