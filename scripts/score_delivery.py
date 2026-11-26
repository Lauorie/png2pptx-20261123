#!/usr/bin/env python3
"""Score the slides of finished runs against their source figures.

Usage: score_delivery.py RUN_DIR [RUN_DIR ...] [--json OUT]

Prints one row per figure and run (recall, precision, colour, overall, see
full_recovery/fidelity.py). With several runs of the same figures it also
prints, per figure, the spread of the overall score between runs.
"""
import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_recovery.fidelity import Fidelity, score  # noqa: E402

logger = logging.getLogger(__name__)


def source_of(run: Path, case: str) -> Optional[Path]:
    """Return the source figure of ``case``, or ``None`` when it is gone."""
    candidate = Path(str(run) + '.input') / 'images' / (case + '.png')
    return candidate if candidate.is_file() else None


def score_run(run: Path) -> Dict[str, Optional[Fidelity]]:
    """Score every figure the run was given; ``None`` marks a figure without a slide."""
    cases = sorted(p.stem for p in (Path(str(run) + '.input') / 'images').glob('*.png'))
    result: Dict[str, Optional[Fidelity]] = {}
    for case in cases:
        preview = run / 'delivery' / case / 'preview.png'
        source = source_of(run, case)
        if source is None or not preview.is_file():
            result[case] = None
            continue
        with Image.open(source) as first, Image.open(preview) as second:
            result[case] = score(first, second)
    return result


def main() -> int:
    """Print the score table and optionally save it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    table: Dict[str, Dict[str, Optional[Fidelity]]] = {run.name: score_run(run) for run in args.runs}
    for name, cases in table.items():
        logger.info('%s', name)
        done: List[float] = []
        for case, value in cases.items():
            if value is None:
                logger.info('  %-18s no slide', case)
                continue
            done.append(value.overall)
            logger.info('  %-18s recall %.3f  precision %.3f  colour %.3f  defects %.3f  overall %.3f',
                        case, value.recall, value.precision, value.colour, value.defects, value.overall)
        if done:
            logger.info('  %-18s slides %d/%d  mean overall %.3f  worst %.3f', 'TOTAL', len(done), len(cases),
                        statistics.mean(done), min(done))
    if len(table) > 1:
        logger.info('spread between runs (overall)')
        for case in sorted({c for cases in table.values() for c in cases}):
            values = [cases[case].overall for cases in table.values() if cases.get(case) is not None]
            if len(values) > 1:
                logger.info('  %-18s mean %.3f  min %.3f  max %.3f  range %.3f', case, statistics.mean(values),
                            min(values), max(values), max(values) - min(values))
    if args.json:
        args.json.write_text(json.dumps({name: {case: None if v is None else v.__dict__ for case, v in cases.items()}
                                         for name, cases in table.items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
