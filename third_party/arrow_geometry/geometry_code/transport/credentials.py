import os
import stat
import fcntl
import json
from pathlib import Path


def load_credentials(base):
    ring = os.environ.get('ARROW_API_KEYRING_FILE')
    if ring:
        path = Path(ring)
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError('Credential file must be private (chmod 600)')
        keys = json.loads(path.read_text())
        if not isinstance(keys, list) or not keys or any(
                not isinstance(k, str) or not k or any(c.isspace() for c in k) for k in keys):
            raise ValueError('Missing or malformed API credential ring')
        # Each transport attempt selects once, shared by all case processes.
        counter = Path(os.environ.get('ARROW_API_KEYRING_COUNTER', str(path)+'.counter'))
        fd = os.open(counter, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'r+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            sequence = int(stream.read() or '0')
            stream.seek(0); stream.write(str(sequence+1)); stream.truncate(); stream.flush()
        index = sequence % len(keys)
        return base.rstrip('/'), keys[index], {
            'source': 'private credential ring outside repository', 'key_saved': False,
            'rotation': 'round_robin', 'key_slot': index+1, 'key_count': len(keys),
            'request_sequence': sequence}
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
