"""Freeze a new experiment snapshot; running processes never import edited source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    project = experiment.parents[1]
    target = args.run_dir / 'revisions' / args.revision
    target.mkdir(exist_ok=False)
    for p in experiment.glob('*.py'):
        ast.parse(p.read_text(), str(p))
        shutil.copy2(p, target / p.name)
    for name in ['src', 'vendor']:
        shutil.copytree(project / name, target / 'reference' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(args.config or experiment / 'settings.json', target / 'settings.json')
    for name in ['DESIGN.zh-CN.md', 'PREFLIGHT.json']:
        if (experiment / name).exists():
            shutil.copy2(experiment / name, target / name)
    hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in target.rglob('*') if p.is_file()}
    (target / 'code_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
