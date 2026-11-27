#!/usr/bin/env python3
"""Score the text of finished runs' slides against their source figures.

Usage: score_text.py RUN_DIR [RUN_DIR ...] [--json OUT] [--differences]

A vision model other than the one that recognises text during the conversion
reads each source figure and its slide, each at three sizes, keeping the
reading the others agree with most (full_recovery/text_reader.py). Prints, per figure, the share
of the source's characters the slide has (recall), the share of the slide's
characters the source has (precision), and the number of words that differ;
``--differences`` lists those words. Positions are not compared.

Uses PNG2PPTX_API_BASE_URL and PNG2PPTX_API_KEY_FILE, like a conversion job.
"""
import argparse
import dataclasses
import json
import logging
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_recovery.text_fidelity import TextFidelity, score_text  # noqa: E402
from full_recovery.text_reader import ReaderConfig, ReadingFailed, consensus, make_reader  # noqa: E402

logger = logging.getLogger(__name__)
Reader = Callable[[Image.Image], List[str]]
Table = Dict[str, Dict[str, Optional[TextFidelity]]]


def _score(source: Path, preview: Path, read: Reader) -> Optional[TextFidelity]:
    if not preview.is_file():
        return None
    try:
        with Image.open(source) as first, Image.open(preview) as second:
            return score_text(first, second, read)
    except ReadingFailed as error:
        logger.error('%s could not be read: %s', source.stem, error)
        return None


def score_run(run: Path, read: Reader, workers: int = 8) -> Dict[str, Optional[TextFidelity]]:
    """Score every figure the run was given; ``None`` marks a figure without a slide or reading."""
    images = Path(str(run) + '.input') / 'images'
    sources = sorted(images.glob('*.png'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(lambda s: _score(s, run / 'delivery' / s.stem / 'preview.png', read), sources))
    return {source.stem: value for source, value in zip(sources, scores)}


def save(table: Table, path: Path) -> None:
    """Write the scores and the differing places of every run as JSON."""
    path.write_text(json.dumps({name: {case: None if value is None else dataclasses.asdict(value)
                                       for case, value in cases.items()}
                                for name, cases in table.items()}, indent=2, ensure_ascii=False))


def report(table: Table, differences: bool) -> None:
    """Log one row per figure and run, and the totals of each run."""
    for name, cases in table.items():
        logger.info('%s', name)
        done = [value for value in cases.values() if value is not None]
        for case, value in cases.items():
            if value is None:
                logger.info('  %-18s no slide or no reading', case)
                continue
            logger.info('  %-18s recall %.3f  precision %.3f  text %.3f  differ %3d  characters %5d',
                        case, value.recall, value.precision, value.overall, len(value.differences),
                        value.source_characters)
            for item in value.differences if differences else ():
                logger.info('      %-30s | %s', item.source[:30] or '(nothing)', item.slide[:30] or '(nothing)')
        if done:
            logger.info('  %-18s slides %d/%d  mean text %.3f  worst %.3f  differ %d', 'TOTAL', len(done), len(cases),
                        statistics.mean(v.overall for v in done), min(v.overall for v in done),
                        sum(len(v.differences) for v in done))


def main() -> int:
    """Print the text score table and optionally save it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--differences', action='store_true')
    parser.add_argument('--model', default='google/gemini-3.1-flash-image')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    config = ReaderConfig(base_url=os.environ['PNG2PPTX_API_BASE_URL'], model=args.model,
                          key=Path(os.environ['PNG2PPTX_API_KEY_FILE']).read_text().strip())
    read = consensus(make_reader(config))
    table: Table = {run.name: score_run(run, read) for run in args.runs}
    report(table, args.differences)
    if args.json:
        save(table, args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
