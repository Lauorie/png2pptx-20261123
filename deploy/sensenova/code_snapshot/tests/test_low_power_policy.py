import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from low_power_policy import LowPowerPolicy


class LowPowerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.policy = LowPowerPolicy(40, 600, lambda: self.now)

    def test_waits_full_ten_minutes_after_cold_start(self):
        self.assertFalse(self.policy.observe(15)['idle_load_enabled'])
        self.now = 599.99
        self.assertFalse(self.policy.observe(20)['idle_load_enabled'])
        self.now = 600
        self.assertTrue(self.policy.observe(20)['idle_load_enabled'])

    def test_power_at_or_above_forty_resets_continuous_low_timer(self):
        self.policy.observe(20)
        self.now = 599
        self.policy.observe(40)
        self.now = 600
        self.policy.observe(39)
        self.now = 1199
        self.assertFalse(self.policy.observe(20)['idle_load_enabled'])
        self.now = 1200
        self.assertTrue(self.policy.observe(20)['idle_load_enabled'])

    def test_high_power_alone_never_starts_idle_load(self):
        self.policy.observe(60)
        self.now = 3600
        self.assertFalse(self.policy.observe(60)['idle_load_enabled'])

    def test_own_power_rise_does_not_turn_idle_load_off(self):
        self.policy.observe(20)
        self.now = 600
        self.assertTrue(self.policy.observe(20)['idle_load_enabled'])
        self.now = 601
        self.assertTrue(self.policy.observe(62)['idle_load_enabled'])

    def test_request_stops_idle_load_and_restarts_observation(self):
        self.policy.observe(20)
        self.now = 600
        self.policy.observe(20)
        self.assertFalse(self.policy.observe(60, 'arrival', request_active=True)['idle_load_enabled'])
        self.now = 900
        self.policy.observe(20, 'arrival', request_active=True)
        self.now = 1000
        self.policy.observe(15, 'finished', can_activate=False)
        self.now = 1599
        self.assertFalse(self.policy.observe(15, 'finished', can_activate=False)['idle_load_enabled'])
        self.now = 1600
        self.assertTrue(self.policy.observe(15, 'finished')['idle_load_enabled'])

    def test_low_observation_overlaps_response_cooldown_without_double_wait(self):
        self.policy.observe(20, 'finished', can_activate=False)
        self.now = 600
        self.assertTrue(self.policy.observe(20, 'finished')['idle_load_enabled'])

    def test_model_startup_does_not_count_as_idle_observation(self):
        self.policy.observe(15, startup_blocked=True)
        self.now = 2000
        self.assertFalse(self.policy.observe(15)['idle_load_enabled'])
        self.now = 2600
        self.assertTrue(self.policy.observe(15)['idle_load_enabled'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
