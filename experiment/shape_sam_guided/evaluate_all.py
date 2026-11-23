"""Complete the planned comparisons sequentially and release the shared model."""
import argparse
from pathlib import Path
import sys
import time

from core import read, save
import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--worker-dir', type=Path, required=True)
    args = parser.parse_args()
    completed, errors = [], []
    for entry in read(args.plan):
        print('COMPARISON_START', entry['run_dir'], flush=True)
        sys.argv = [report.__file__, '--run-dir', entry['run_dir'], '--config', entry['config']]
        try:
            report.main()
            completed.append(entry['run_dir'])
        except Exception as exc:
            errors.append({'run_dir': entry['run_dir'], 'error': str(exc)})
            print('COMPARISON_FAILED', errors[-1], flush=True)
        save(args.worker_dir / 'evaluation_batch_progress.json',
             {'completed': completed, 'errors': errors, 'updated_at': time.time()})
    if errors:
        raise RuntimeError('Some comparisons failed; cached completed reviews are retained for recovery.')
    save(args.worker_dir / 'STOP.json', {'reason': 'All planned reconstruction and evaluation completed.',
         'reports': [str(Path(p) / 'aggregate.json') for p in completed], 'completed_at': time.time()})
    print('ALL_INFERENCE_COMPLETE_WORKER_STOP_REQUESTED', flush=True)


if __name__ == '__main__':
    main()
