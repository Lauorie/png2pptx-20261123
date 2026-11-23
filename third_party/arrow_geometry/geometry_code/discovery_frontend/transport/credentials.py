import os
import stat
from pathlib import Path


def load_credentials(base):
    key = os.environ.get('ARROW_API_KEY') or os.environ.get('BOYUE_API_KEY')
    source = 'environment'
    if not key:
        path = Path(os.environ.get('ARROW_API_KEY_FILE', '/home/yinjinhui/.config/arrow_partition_points/boyue.key'))
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError('Credential file must be private (chmod 600)')
        key = path.read_text().strip()
        source = 'private credential file outside repository'
    if not key or any(c.isspace() for c in key):
        raise ValueError('Missing or malformed API credential')
    return base.rstrip('/'), key, {'source': source, 'key_saved': False}
