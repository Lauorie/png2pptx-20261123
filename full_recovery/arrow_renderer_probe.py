"""Let the arrow renderer's version probe survive a slow start.

The frozen arrow code asks the renderer for its version once per process and
allows 15 seconds. The renderer answers in about a second, but on a machine
whose CPU quota is used up its start has taken longer, and that one timeout
failed every arrow of the image and with it the whole job.
"""
from pathlib import Path
from typing import Tuple

# Seconds allowed for each try.
PROBE_TIMEOUTS: Tuple[int, ...] = (15, 60, 180)
_ANCHOR = ("                version = subprocess.run(command, capture_output=True, text=True, check=True, "
           "timeout=15).stdout.strip()\n")
_RETRY = (
    "                for attempt, seconds in enumerate(PROBE_TIMEOUTS):\n"
    "                    try:\n"
    "                        version = subprocess.run(command, capture_output=True, text=True, check=True,\n"
    "                                                 timeout=seconds).stdout.strip()\n"
    "                        break\n"
    "                    except subprocess.TimeoutExpired:\n"
    "                        if attempt + 1 == len(PROBE_TIMEOUTS):\n"
    "                            raise\n")
_HEAD_ANCHOR = '_LOCK = threading.Lock()\n'


def install(geometry_code: Path) -> None:
    """Make the version probe of one prepared run try again when the renderer is slow to answer.

    Args:
        geometry_code: The run's own copy of the arrow ``geometry_code`` directory.

    Raises:
        ValueError: When the probe does not have the expected, unpatched form.
    """
    module = Path(geometry_code)/'native_runtime.py'
    source = module.read_text()
    if source.count(_ANCHOR) != 1 or source.count(_HEAD_ANCHOR) != 1 or 'PROBE_TIMEOUTS' in source:
        raise ValueError('Unexpected arrow renderer probe source: ' + str(module))
    source = source.replace(_HEAD_ANCHOR, _HEAD_ANCHOR + 'PROBE_TIMEOUTS = ' + repr(PROBE_TIMEOUTS) + '\n')
    module.write_text(source.replace(_ANCHOR, _RETRY))
