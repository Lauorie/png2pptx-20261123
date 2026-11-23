"""Text-only schema repair with a conservative semantic preservation guard."""
from collections import Counter
import json
import re

SYSTEM = ('你是JSON格式修复器。只修复给定回复的语法、容器、字段类型和明确的引用格式。'
          '没有图片，不重新识别，不猜测对象，不改变坐标、对象数量、身份含义或排除决定。'
          '若缺少语义内容或引用无法唯一确定，返回{"format_repair_failed":true}。只输出JSON。')


def prompt_for(original, schema_prompt, error):
    if isinstance(schema_prompt, str):
        schema_prompt = {'output_schema': schema_prompt}
    return {'task':'修复原回复的JSON格式，使其满足校验；保持原有有效内容。',
            'original_reply':original,'validation_error':str(error),
            'schema':schema_prompt.get('output_schema',schema_prompt.get('response_format',{})),
            'focus_ids':[o['id'] for o in schema_prompt.get('focus',[])],
            'target_size_px':schema_prompt.get('target_size_px'),
            'rules':['不增加或删除对象，不改变bbox等数值、对象种类、完整性或语义描述。',
                     '可以补语法符号、修数组/对象容器、将字面量布尔/数字字符串转换为相同值。',
                     '同一focus已在objects中，而removed仅说明框内另一些非目标部件（圆轮廓/文字/背景等）时，可将这些原记录原封不动移入context_notes数组，removed留空；保留全部说明文字与ID，不删除objects。若removed确实否定objects中的同一对象则不可自行消除矛盾。',
                     '缺少对象判断、缺少坐标或引用不明确时不要猜，返回format_repair_failed。']}


def scalar(value):
    if isinstance(value,str):
        if value in ('true','false','null'):return value
        try:
            number=float(value)
            if number==number and abs(number)!=float('inf'):return 'number:'+repr(number)
        except ValueError:pass
        return 'string:'+value
    if value is None:return 'null'
    if isinstance(value,bool):return str(value).lower()
    if isinstance(value,(int,float)):return 'number:'+repr(float(value))
    raise ValueError('Not a scalar')


def leaves(value):
    if isinstance(value,dict):return sum((leaves(v) for v in value.values()),[])
    if isinstance(value,list):return sum((leaves(v) for v in value),[])
    return [scalar(value)]


def lexical_values(text):
    # Only punctuation repairs are provable for unparseable JSON. Unclosed
    # quotes/ambiguous text fail this guard and fall back to a new visual call.
    tokens=list(re.finditer(r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b',text))
    values=[]
    for m in tokens:
        token=m.group();tail=text[m.end():].lstrip()
        if token.startswith('"') and tail.startswith(':'):continue
        values.append(scalar(json.loads(token)))
    return values


def records(value):
    out=[]
    if isinstance(value,dict):
        if any(k.startswith('bbox') for k in value):out.append(value)
        else:
            for v in value.values():out.extend(records(v))
    elif isinstance(value,list):
        for v in value:out.extend(records(v))
    return out


def canonical(value):
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    return scalar(value)


def preservation_audit(original_text, repaired):
    if isinstance(repaired,dict) and repaired.get('format_repair_failed'):
        raise ValueError('Text repair reports missing or ambiguous semantic content')
    try:original=json.loads(original_text.strip().removeprefix('```json').removesuffix('```').strip())
    except (ValueError,AttributeError):original=None
    before=Counter(leaves(original)) if original is not None else Counter(lexical_values(original_text))
    after=Counter(leaves(repaired))
    if before!=after:raise ValueError('Format repair changed scalar values or object content')
    if original is None and lexical_values(original_text)!=leaves(repaired):
        raise ValueError('Cannot prove field associations after reordering malformed JSON')
    if original is not None:
        old=records(original);new=records(repaired)
        if len(old)!=len(new):raise ValueError('Format repair changed object count')
        # Values may not be swapped between existing objects, even if the
        # scalar multiset for the full response happens to stay the same.
        if Counter(json.dumps(canonical(o),sort_keys=True) for o in old)!=Counter(json.dumps(canonical(o),sort_keys=True) for o in new):
            raise ValueError('Format repair changed per-object fields or assignments')
    return {'scalar_values_preserved':True,'per_object_fields_checked':original is not None,
            'images_sent':0,'unparseable_guard':'lexical scalar inventory' if original is None else None}
