"""Publish immutable render snapshots through one atomic directory pointer."""
import os
from pathlib import Path
from ..common import read, sha, save, now

FILES = ('styled.svg', 'styled.png', 'styled.pptx', 'native.pdf', 'native.png',
         'comparison.png', 'export.json', 'calibration.json', 'live.json')


def publish(folder, snapshot, sequence):
    folder=Path(folder);snapshot=Path(snapshot)
    manifest=read(snapshot/'export.json')
    if not manifest.get('native_rendered'):
        raise ValueError('Live publication requires a complete native PPT render')
    for name in FILES[:6]:
        if manifest['files'].get(name)!=sha(snapshot/name):
            raise ValueError('Incomplete or changed live snapshot: '+name)
    save(snapshot/'live.json',{'snapshot':str(snapshot.relative_to(folder)),
                             'sequence':sequence,'styles_sha256':sha(snapshot/'styles.json'),
                             'published_utc':now(),'visual_accuracy_verified':False})
    # Existing public paths all resolve via `current`; readers keep the old valid
    # version throughout rendering and switch only after every output is complete.
    for name in FILES:
        link=folder/name;target='current/'+name
        if link.is_symlink() and os.readlink(link)==target:continue
        temporary=folder/('.'+name+'.link.tmp')
        temporary.unlink(missing_ok=True);temporary.symlink_to(target);temporary.replace(link)
    temporary=folder/'.current.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(snapshot.relative_to(folder),target_is_directory=True)
    temporary.replace(folder/'current')
