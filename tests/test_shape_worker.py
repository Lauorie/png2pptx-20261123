"""Exercise the actual worker supervisor with fake processes, never GPU jobs."""
from pathlib import Path
import fcntl
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'experiment/shape_sam_guided'
sys.dont_write_bytecode = True
with patch.object(sys, 'path', [str(HERE), *sys.path]):
    spec = importlib.util.spec_from_file_location('shape_guard_worker', HERE/'worker.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'project_root': str(ROOT), 'sam_cache_only': True, 'glm_tp': 8,
                       'glm_model': 'test-model', 'glm_url': 'unused', 'sam_threshold': .3,
                       'training_run': str(self.root/'training'), 'baseline_root': 'unused',
                       'baseline_run': 'unused', 'sam_ports': [18103, 18104, 18105, 18106]}
        for rel in ['code/reference', 'comparison', 'shared/sam_cache']:
            (self.root/rel).mkdir(parents=True)
        save(self.root/'settings.json', self.config)
        save(self.root/'shared/sam_precompute.json', {'completed': 1, 'expected': 1})
        save(self.root/'comparison/aggregate.json', {})
        self.processes = []; self.preflights = []; self.fail_first_forward = False

    def run_worker(self):
        outer = self
        class Process:
            def __init__(self, argv, **kwargs):
                self.argv, self.env = argv, kwargs['env']
                self.pid = 8000000+len(outer.processes)
                script = next((Path(v).name for v in argv if str(v).endswith('.py')), '')
                self.kind = script
                self.returncode = None if script in {'sam_service.py', 'serve_glm.py'} else 0
                if script == 'run.py' and outer.fail_first_forward and not any(p.kind == 'run.py' for p in outer.processes):
                    self.returncode = 1
                if script == 'sam_service.py':
                    save(Path(argv[argv.index('--state')+1]), {'phase': 'ready'})
                if script == 'serve_glm.py':
                    # Enforce exclusive GPU phases, including the release wait.
                    outer.assertTrue(all(p.poll() is not None for p in outer.processes if p.kind == 'sam_service.py'))
                    save(Path(argv[argv.index('--run-dir')+1])/'service-state.json', {'phase': 'ready'})
                outer.processes.append(self)
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.returncode = 0
                return 0
        def preflight(argv, **kwargs):
            self.preflights.append((argv, kwargs['env']))
        with patch.object(sys, 'argv', ['worker', '--run-dir', str(self.root)]), \
             patch.object(worker.subprocess, 'Popen', Process), \
             patch.object(worker.subprocess, 'run', preflight), \
             patch.object(worker.signal, 'signal'), patch.object(worker.os, 'killpg'):
            worker.main()

    def test_cached_sam_does_not_launch_gpu_sam_services(self):
        self.run_worker()
        self.assertFalse(any(p.kind == 'sam_service.py' for p in self.processes))
        glm = next(p for p in self.processes if p.kind == 'serve_glm.py')
        self.assertEqual(glm.env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(glm.argv[glm.argv.index('--tp')+1], '8')
        self.assertEqual(json.loads((self.root/'status.json').read_text())['phase'], 'completed')

    def test_uncached_sam_releases_devices_before_glm_starts(self):
        self.config['sam_cache_only'] = False; save(self.root/'settings.json', self.config)
        self.run_worker()
        services = [p for p in self.processes if p.kind == 'sam_service.py']
        self.assertEqual([p.env['CUDA_VISIBLE_DEVICES'] for p in services], ['4', '5', '6', '7'])
        self.assertTrue(all(p.poll() is not None for p in services))
        self.assertEqual(len([p for p in self.processes if p.kind == 'serve_glm.py']), 1)

    def test_resume_binds_forward_and_preflight_to_the_same_new_reference(self):
        self.fail_first_forward = True
        code = self.root/'revisions/r02'; (code/'reference').mkdir(parents=True)
        for rel in ['native_review', 'accepted']:
            (self.root/'comparison_r02'/rel).mkdir(parents=True)
        save(code/'settings.json', self.config)
        save(self.root/'comparison_r02/aggregate.json', {})
        save(self.root/'RESUME.json', {'revision': 'r02', 'comparison': 'comparison_r02'})
        self.run_worker()
        forwards = [p for p in self.processes if p.kind == 'run.py']
        self.assertEqual(len(forwards), 2)
        self.assertEqual(forwards[0].env['SHAPE_REFERENCE_ROOT'], str(self.root/'code/reference'))
        self.assertEqual(forwards[1].env['SHAPE_REFERENCE_ROOT'], str(code/'reference'))
        self.assertEqual(self.preflights[-1][1]['SHAPE_REFERENCE_ROOT'], str(code/'reference'))
        self.assertEqual(next(p for p in self.processes if p.kind == 'report.py').env['SHAPE_REFERENCE_ROOT'], str(code/'reference'))

    def test_competing_supervisor_cannot_start_or_overwrite_status(self):
        save(self.root/'status.json', {'owner': 'other worker'})
        with (self.root/'.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_worker()
        self.assertFalse(self.processes)
        self.assertEqual(json.loads((self.root/'status.json').read_text()), {'owner': 'other worker'})

    def test_single_arm_progress_uses_its_configured_arm(self):
        comparison = self.root/'comparison'
        save(comparison/'manifest.json', {'images': [{'id': 'sample'}], 'config': {'arms': ['sam_fixed_base']}})
        save(comparison/'sam_fixed_base/cases/sample/forward_complete.json', {})
        self.assertEqual(worker.completion_counts(comparison),
                         {'forward_pairs_complete': 1, 'forward_arms_complete': 1, 'forward_pairs_expected': 1})


if __name__ == '__main__':
    unittest.main()
