"""Keep review PNGs inside the report directory without changing raw evidence."""
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path


PNG_LINK = re.compile(r'(!?\[[^\]\n]*\]\()([^\n)]+\.png)(\))')
REPORTS = ('独立身份与只读参考_16例.md', '只读参考入口.md')


def asset_relative_path(source, run):
    if source.is_relative_to(run):
        return Path('assets') / source.relative_to(run)
    if source.is_relative_to(run.parent):
        return Path('assets/source_runs') / source.relative_to(run.parent)
    source_key = hashlib.sha256(str(source).encode('utf-8')).hexdigest()[:16]
    return Path('assets/external') / source_key / source.name


def _replace_file(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def localize_pngs(text, run, report):
    def replace(match):
        source = Path(match[2])
        if not source.is_absolute():
            return match[0]
        relative = asset_relative_path(source, run)
        target = report / relative
        source_stat = source.stat()
        target_stat = target.stat() if target.is_file() else None
        if target_stat is None or (source_stat.st_size, source_stat.st_mtime_ns) != (target_stat.st_size, target_stat.st_mtime_ns):
            _replace_file(target, lambda temporary: shutil.copy2(source, temporary))
        return match[1] + relative.as_posix() + match[3]

    return PNG_LINK.sub(replace, text)


def write_report(path, text, run):
    updated = localize_pngs(text, run, path.parent)
    _replace_file(path, lambda temporary: temporary.write_text(updated, encoding='utf-8'))


def repair_reports(run, report):
    """Also repair interim reports published by the frozen running experiment."""
    for name in REPORTS:
        path = report / name
        if not path.is_file():
            continue
        original = path.read_bytes()
        updated = localize_pngs(original.decode('utf-8'), run, report).encode('utf-8')
        if updated != original and path.read_bytes() == original:
            _replace_file(path, lambda temporary: temporary.write_bytes(updated))
