"""Independent endpoint recovery after the existing batch style review."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import head_style_prompt
import style_contract
from simple_style_contract import endpoint, category
from transport.common import write_json as write


def validate(answer, path_id):
    if not isinstance(answer, dict) or answer.get('id') != path_id:
        raise ValueError('Return the requested path ID')
    if set(answer) != {'id', 'start', 'end'}:
        raise ValueError('Return only id, start and end')
    if any(answer[end] not in ('none', 'circle', 'arrow') for end in ('start', 'end')):
        raise ValueError('Endpoint must be none, circle or arrow')
    return answer


def apply_prediction(row, answer, request_dir):
    result = copy.deepcopy(row)
    validate(answer, row['id'])
    style = result['style']
    previous_review = {key: copy.deepcopy(style[key]) for key in ('state', 'needs_review', 'review') if key in style}
    result['pre_head_style_review'] = previous_review
    for end in ('start', 'end'):
        style[end] = endpoint(answer[end])
        result[end+'_head'] = answer[end] != 'none'
        result[end+'_head_type'] = style[end]['type'] if style[end] else None
    result['heads'] = [result[end+'_head_type'] or 'none' for end in ('start', 'end')]
    result['head_source'] = head_style_prompt.VERSION
    result['head_recovery'] = dict(state='estimated', critic_run=False,
                                   prompt_version=head_style_prompt.VERSION,
                                   request=str(request_dir.resolve()))
    style.pop('review', None)
    style.update(state='estimated', needs_review=True)
    return result


def recover(rows, input_dir, layout, out, config, model, size):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    input_dir = Path(input_dir)
    before = copy.deepcopy(rows)
    write(out/'before_rows.json', before)
    targets = [row for row in rows if row['kind'] != 'block_arrow']
    if not targets:
        write(out/'status.json', dict(state='skipped', reason='no ordinary paths'))
        return before, []
    tags = {path_id: tag for tag, path_id in layout['locator_paths'].items()}
    images = [('SOURCE ORIGINAL: original image evidence', input_dir/'clean.png'),
              ('LOCATORS: annotations only', input_dir/'locators.png')]
    if config.get('head_recovery_reference', True):
        images.append(('REFERENCE EXAMPLES: generic none/circle/arrow, not the target',
                       Path(__file__).parent/'assets/head_reference/head_reference.png'))
    predictions = {}; issues = []; records = []

    def work(row):
        tag = tags[row['id']]
        request = out/'paths'/tag
        answer = model.ask(request, head_style_prompt.prompt(row, tag, size), images,
                           lambda value: validate(value, row['id']), system=head_style_prompt.SYSTEM)
        return apply_prediction(row, answer, request), answer

    with ThreadPoolExecutor(max_workers=min(len(targets), config.get('head_recovery_workers', 4))) as pool:
        jobs = {pool.submit(work, row): row for row in targets}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                updated, answer = future.result()
                predictions[row['id']] = updated
                records.append(dict(id=row['id'], state='completed', before={end:category(row['style'][end]) for end in ('start','end')}, prediction=answer))
            except Exception as exc:
                issue = dict(stage='head_recovery', id=row['id'], error=str(exc), retained='previous heads')
                issues.append(issue)
                records.append(dict(id=row['id'], state='failed', error=str(exc), retained='previous heads'))
            write(out/'progress.json', dict(ended=len(records), total=len(targets), paths=records))
    result = [predictions.get(row['id'], copy.deepcopy(row)) for row in rows]
    assert style_contract.geometry_signature(before) == style_contract.geometry_signature(result)
    for old, new in zip(before, result):
        for key in ('color', 'width_px', 'dash'):
            assert old['style'].get(key) == new['style'].get(key), 'Head recovery changed shaft style'
        if old['kind'] == 'block_arrow':
            assert old == new, 'Head recovery changed block arrow'
    write(out/'rows.json', result)
    write(out/'status.json', dict(state='completed_with_issues' if issues else 'completed',
          paths=len(targets), failed=len(issues), issues=issues, geometry_unchanged=True,
          shaft_style_unchanged=True, critic_run=False, prompt_version=head_style_prompt.VERSION))
    return result, issues
