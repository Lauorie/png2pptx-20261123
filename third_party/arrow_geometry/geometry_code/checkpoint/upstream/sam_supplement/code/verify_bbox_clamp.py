"""Replay complete saved replies, comparing bounding with the prior adapter. No API."""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from supplement_protocol import MissingProtocol, ReviewProtocol
from transport.common import write_json


def run(source, output):
    source = source.resolve()
    output = output.resolve()
    spec = importlib.util.spec_from_file_location('old_supplement_protocol',
        source/'receiver_v2/code/supplement_protocol.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    stats = Counter(); recovered = []; unresolved = []; failures = []
    for stage, pattern in [('missing','*/proposal_views/*/view_*/request'),
                           ('review','*/review/round_*/B*/view_*/request')]:
        for request in sorted((source/stage/'cases').glob(pattern)):
            call = request/'call_000'
            if not (call/'status.json').exists():
                continue
            status = json.loads((call/'status.json').read_text())
            if status.get('state') != 'transport_complete' or not status.get('sse_done'):
                continue
            prompt = json.loads((request/'prompt.json').read_text())['prompt']
            args = (prompt['input_size'],) if stage == 'missing' else (prompt['input_size'],prompt['editable_additions'])
            new_contract = (MissingProtocol if stage == 'missing' else ReviewProtocol)(*args)
            old_contract = (old.MissingProtocol if stage == 'missing' else old.ReviewProtocol)(*args)
            try:
                raw = new_contract.parse((call/'content.txt').read_text())
                before = old_contract(raw); after = new_contract(raw)
                assert after['_protocol']['raw_answer'] == raw
                assert new_contract.validate_saved(after) == after
                stats[stage+'_requests'] += 1
                stats[stage+'_clamped_boxes'] += len(after['_protocol']['bbox_adjustments'])
                if stage == 'missing':
                    previous = {a['source_index']:a['accepted_bbox'] for a in before['_protocol']['accepted']}
                    current = {a['source_index']:a['accepted_bbox'] for a in after['_protocol']['accepted']}
                    assert all(current[k] == box for k,box in previous.items())
                    stats['unchanged_previously_valid_boxes'] += len(previous)
                    stats['recovered_missing_candidates'] += len(current)-len(previous)
                else:
                    old_ops = {(a['operation'],a['kind']) for a in before['_protocol']['accepted']}
                    new_ops = {(a['operation'],a['kind']) for a in after['_protocol']['accepted']}
                    assert old_ops <= new_ops
                    assert before.get('keep_ids') == after.get('keep_ids') and before.get('discard') == after.get('discard')
                    current = {tuple(p['source_ids']):p for p in after.get('replacements',[])}
                    for patch in before.get('replacements',[]):
                        assert current[tuple(patch['source_ids'])] == patch
                        stats['unchanged_previously_valid_boxes'] += len(patch['regions'])
                    stats['recovered_review_operations'] += len(new_ops)-len(old_ops)
                if after['_protocol']['bbox_adjustments']:
                    record=dict(request=str(request),stage=stage,normalized=after)
                    recovered.append(record)
                    write_json(output/'recovered'/f'{len(recovered):03d}.json',record)
                for bad in after['_protocol']['rejected']:
                    unresolved.append(dict(request=str(request),reason=bad['reason']))
            except Exception as exc:
                failures.append(dict(request=str(request),error=repr(exc)))
    report=dict(api_calls=0,counts=dict(stats),raw_answers_preserved=True,
                recovered_requests=[dict(request=x['request'],stage=x['stage'],
                    adjustments=x['normalized']['_protocol']['bbox_adjustments']) for x in recovered],
                remaining_non_boundary_issues=unresolved,failures=failures)
    write_json(output/'checks.json',report)
    lines=['# 越界坐标截断：同轮回复离线回放','',
           '仅重新解析完整回复，API调用0；原始坐标保留，先截断到当前输入图范围，再由后台换算回原图。','',
           '统计：`'+json.dumps(dict(stats),ensure_ascii=False)+'`。','',
           '已有有效框保持不变；裁断后无面积、未知ID和冲突操作不猜测处理。此处通过表示协议接收，不等于视觉正确。','',
           '|阶段|请求|调整|','|---|---|---|']
    for x in recovered:
        edits='; '.join(str(a['raw_bbox'])+' → '+str(a['bounded_bbox']) for a in x['normalized']['_protocol']['bbox_adjustments'])
        lines.append(f"|{x['stage']}|[{Path(x['request']).relative_to(source)}]({x['request']})|{edits}|")
    (output/'越界回放.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['recovered_requests','remaining_non_boundary_issues']},ensure_ascii=False))
    if failures:
        raise AssertionError(failures)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
