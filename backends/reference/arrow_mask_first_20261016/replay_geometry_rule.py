"""A single saved geometry request, changing only an explicit prompt rule."""
import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from PIL import Image
import protocol
import segmented_geometry as sg
import style_contract
from lib import native, viz
from transport.common import read_json as read, write_json as write, sha256 as sha
from transport.model import Model
from run_region_ablation import usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    out = args.run.resolve()
    lock = (out / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(out / 'code_manifest.json')
    if any(sha(out / 'code' / n) != h for n, h in manifest['files'].items()):
        raise ValueError('Frozen code changed')
    if sha(out / 'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen config changed')
    spec = read(out / 'selection.json')
    source = Path(spec['source_geometry'])
    q = read(source / 'prompt.json')
    original_q = read(source / 'prompt.json')
    q['instructions'].append(spec['additional_instruction'])
    images = [(r['label'], Path(r['path'])) for r in read(source / 'input/images.json')]
    baseline = read(source / 'baseline_metadata.json')
    for row in baseline.values():
        row['source_geometry_request'] = row['geometry_request']
        row['geometry_request'] = str(out / 'request')
    write(out / 'prompt.json', q)
    write(out / 'input_audit.json', dict(source=str(source), source_prompt_sha256=sha(source / 'prompt.json'),
                                       images=[dict(label=l, path=str(f), sha256=sha(f)) for l, f in images],
                                       same_targets=q['targets'] == original_q['targets'],
                                       only_prompt_change=spec['additional_instruction']))
    write(out / 'process.json', dict(pid=os.getpid(), started=time.time()))
    started = time.monotonic()
    state = dict(state='running')
    write(out / 'progress.json', state)
    try:
        model = Model(read(out / 'config.json'), run_dir=out, system=protocol.SYSTEM)
        answer = model.ask(out / 'request', q, images, lambda a: sg.validate(a, baseline, q['size']))
        write(out / 'answer.json', answer)
        rows = sg.geometry(answer, baseline)
        clean = Image.open(images[0][1]).convert('RGB')
        clean.save(out / 'clean.png')
        for row in rows:
            row.update(parent_id=row['physical_target_id'], heads=['unknown', 'unknown'],
                       start_head=False, end_head=False)
            row['style'] = style_contract.provisional(clean, None, row)
        write(out / 'rows_local.json', rows)
        sg.draw_points(answer, out / 'clean.png', out / 'points.png')
        viz.render(out / 'clean.png', rows, out / 'result')
        native.write(clean.size, rows, [r['style'] for r in rows], out / 'arrows.pptx')
        native.render(out / 'arrows.pptx', out / 'native', clean.size)
        state = dict(state='completed', paths=len(rows), native_rendered=True,
                     style_fitted=False, geometry_programmatically_adjusted=False)
    except Exception as exc:
        state = dict(state='failed', error=str(exc), traceback=traceback.format_exc())
    state.update(seconds=time.monotonic() - started, api=usage(out), accuracy_measured=False)
    write(out / 'finished.json', state)
    write(out / 'progress.json', state)
    print(state['state'], flush=True)


if __name__ == '__main__':
    main()
