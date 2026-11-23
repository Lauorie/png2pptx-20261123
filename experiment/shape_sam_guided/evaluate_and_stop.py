"""Finish the last planned model evaluation, then release the persistent worker."""
import argparse
from pathlib import Path
import sys
import time

from core import save
import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--worker-dir', type=Path, required=True)
    parser.add_argument('--required-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    sys.argv = [str(Path(report.__file__)), '--run-dir', str(args.run_dir), '--config', str(args.config)]
    report.main()
    reports = [*args.required_report, args.run_dir / 'aggregate.json']
    if all(p.exists() for p in reports):
        save(args.worker_dir / 'STOP.json', {'reason': 'All planned inference and evaluation completed.',
             'reports': [str(p) for p in reports], 'completed_at': time.time()})
        print('ALL_INFERENCE_COMPLETE_WORKER_STOP_REQUESTED', flush=True)
    else:
        print('Other required reports are incomplete; worker remains available for recovery.', flush=True)


if __name__ == '__main__':
    main()
