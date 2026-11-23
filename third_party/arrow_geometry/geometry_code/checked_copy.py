"""Buffered, verified, atomic copies for the shared experiment filesystem."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def copy_file(source, destination, *, follow_symlinks=True):
    """copy2-compatible regular-file copy, without sendfile/copy_file_range.

    Publish only after the bytes read from source match the temporary copy.
    A failed copy leaves the previous destination intact.
    """
    source=Path(source);destination=Path(destination)
    if destination.is_dir():destination=destination/source.name
    if destination.exists() and os.path.samefile(source,destination):
        raise shutil.SameFileError(str(source))
    if not follow_symlinks and source.is_symlink():
        raise ValueError('Snapshot copying requires regular file contents')
    expected=hashlib.sha256()
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='.'+destination.name+'.',suffix='.copy',
                                         dir=destination.parent,delete=False) as output:
            temporary=Path(output.name)
            with source.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    expected.update(block);output.write(block)
            output.flush();os.fsync(output.fileno())
        if digest(temporary)!=expected.hexdigest():
            raise OSError('Copied bytes differ from source: '+str(source))
        shutil.copystat(source,temporary,follow_symlinks=follow_symlinks)
        os.replace(temporary,destination)
        return str(destination)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
