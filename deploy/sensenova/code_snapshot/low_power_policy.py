"""Start idle load only after ten continuous idle minutes below 40% power."""
import time


class LowPowerPolicy:
    def __init__(self, threshold_pct=40, duration_seconds=600, clock=None):
        self.threshold = float(threshold_pct)
        self.duration = float(duration_seconds)
        self.clock = clock or time.monotonic
        self.low_since = None
        self.running = False
        self.revision = None

    def observe(self, measured_pct, request_revision=None, request_active=False,
                startup_blocked=False, can_activate=True):
        now = self.clock()
        if request_revision != self.revision:
            self.revision = request_revision
            self.running = False
            self.low_since = None
        if request_active or startup_blocked:
            self.running = False
            self.low_since = None
            phase = 'yielding_to_inference' if request_active else 'waiting_for_service'
        elif self.running:
            # Keep the 60% load on until a request arrives. Its own power rise
            # must not turn it off and cause ten-minute on/off oscillations.
            phase = 'running'
        else:
            if measured_pct < self.threshold:
                if self.low_since is None:
                    self.low_since = now
                if now - self.low_since >= self.duration and can_activate:
                    self.running = True
            else:
                self.low_since = None
            phase = 'running' if self.running else 'monitoring_low_power'
        elapsed = 0.0 if self.low_since is None else now - self.low_since
        return dict(phase=phase, idle_load_enabled=self.running,
                    low_power_threshold_pct=self.threshold,
                    low_power_required_seconds=self.duration,
                    low_power_elapsed_seconds=elapsed)
