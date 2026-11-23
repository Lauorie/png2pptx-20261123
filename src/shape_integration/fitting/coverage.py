"""Whole-image review, image deduplication and discovery queue proposals."""
from copy import deepcopy
from PIL import Image

from ..common import read, save, overlay, mapped_box, valid_box, digest
from ..image_conflicts import run_filter
from ..scope import SHAPE_SCOPE
from .control import schema, validate, SYSTEM
from .coordinates import coordinate_mode, model_size, to_pixels, NORM1000_BOX_CONVENTION


def isolate_repair_parts(raw, scene):
    """Hold a repair with unknown part IDs while retaining independent work."""
    result=deepcopy(raw);kept=[];held=result.pop('held_repairs',[])
    for row in result['repair_existing']:
        available={scene['parts'][k]['id'] for k in scene['objects'][row['object_id']]['part_keys']}
        if any(not set(issue['part_ids'])<=available for issue in row['issues']):
            held.append(row)
        else:kept.append(row)
    result['repair_existing']=kept
    if held:
        result['held_repairs']=held;result['status']='uncertain'
        result['uncertain']=list(dict.fromkeys([*result['uncertain'],
            *('Repair held: unknown part IDs for '+r['object_id'] for r in held)]))
    return result


def review(scene, item, workspace, folder, runtime, options, existing, assets, threshold, round_index):
    folder.mkdir(parents=True, exist_ok=True)
    size = item['input_size']
    mode = coordinate_mode(options)
    objects = [{**o, 'bbox_2d': mapped_box(o['bbox_2d'], item['source_size'], size)} for o in existing]
    with Image.open(workspace/item['input']) as image:
        overlay(image, objects, folder/'inventory.png')
    # Source pixels inside an overlay conceal precisely the missing children
    # this existing coverage stage needs to find.
    from .views import geometry_only
    geometry_only({'parts':list(scene['parts'].values())},item['source_size'],
                  folder/'geometry_original.png',labels=False)
    with Image.open(folder/'geometry_original.png') as image:
        image.resize(tuple(size), Image.Resampling.LANCZOS).save(folder/'fit.png')
    candidate_spec = schema({'label': {'type': 'string', 'minLength': 1}, 'kind': {'type': 'string', 'minLength': 1},
                             'bbox_2d': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}},
                             'reason': {'type': 'string', 'minLength': 1}})
    pair = schema({'object_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2, 'maxItems': 2, 'uniqueItems': True},
                   'reason': {'type': 'string', 'minLength': 1}})
    repair = schema({'object_id': {'type': 'string'}, 'issues': {'type': 'array', 'minItems': 1,
        'items': schema({'kind': {'enum': ['missing', 'extra']},
                         'part_ids': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
                         'evidence': {'type': 'string', 'minLength': 1},
                         'action': {'type': 'string', 'minLength': 1}})}})
    repair['properties']['source_extent']={'type':'array','minItems':4,'maxItems':4,
        'items':{'type':'number'}}
    spec = schema({'status': {'enum': ['complete', 'uncertain']},
                   'new_objects': {'type': 'array', 'items': candidate_spec, 'maxItems': options['max_new_candidates_per_round']},
                   'possible_duplicates': {'type': 'array', 'items': pair},
                   'uncertain': {'type': 'array', 'items': {'type': 'string'}},
                   'repair_existing': {'type': 'array', 'items': repair}})
    ids = {o['id'] for o in existing}
    if ids:
        pair['properties']['object_ids']['items']['enum'] = sorted(ids)
        repair['properties']['object_id']['enum'] = sorted(ids)
    prompt = {'version': 'shape_integration.fit_coverage.12', 'scope': SHAPE_SCOPE,
              'image_size': size, 'object_states': [
                  {'id': o['id'], 'label': o['label'], 'kind': o['kind'],
                   'state': scene['objects'][o['id']]['state'],
                   'branch_ownership': {k:v for k,v in (scene['objects'][o['id']].get('declared_ownership') or {}).items()
                       if k in ('unit','retained_roles','delegated_parts')},
                   'source_registration_conflicts':[
                       {k:r[k] for k in ('reason','parts','observed_parts','source_components') if k in r}
                       for r in scene['objects'][o['id']].get('source_registration_conflicts',[])],
                   'represented_parts':[
                       {'id':scene['parts'][key]['id'],'kind':scene['parts'][key]['kind'],
                        'fitted_bbox_pixels': [round(v,2) for v in mapped_box(
                            scene['parts'][key]['bbox_2d'],item['source_size'],size)],
                        'closed':scene['parts'][key].get('commands',[])[-1:]==[['Z']]}
                       for key in scene['objects'][o['id']]['part_keys'] if scene['parts'][key]['status']=='ok'],
                   'fitted_part_kinds': sorted({scene['parts'][key]['kind'] for key in scene['objects'][o['id']]['part_keys']})}
                  for o in existing],
              'task': '对照原图与白底恢复几何，只补缺失的 shape。候选框仅表示定位，不代表内部已恢复。',
              'rules': [
                  '逐区域按原图从上到下检查，每列、每组直到最底端；重点核对最后一行、文字背后的框、叠层露出的边角和原图边缘。重复列分别建立对应，不因另一列已有同类形状而认为本列已覆盖；也不能按对称性补出原图没有的成员。',
                  '先看 represented_parts 和白底图。已有外框不代表职责范围内的成员已恢复。已有owner缺少固有成员时用repair_existing；没有对应owner的独立对象用new_objects。同一原图成员只登记一次。',
                  '白底图只显示几何线稿；closed=true的面已有可填充区域，缺少颜色不是missing。只在原图区域没有对应闭合轮廓时补面。',
                  'represented_parts的fitted_bbox_pixels使用所给全图的像素坐标，只用于把部件ID定位到当前恢复图，不是原图真值，也不改变输出坐标约定。先从clean source逐一建立实际对应；同类、同名或相邻的部件不能代替另一个缺失实例。',
                  '部件ID中的行列编号、编号缺口和旧回复中的数量不是原图证据。网格先沿原图两组相邻外边逐个确认内部线的交点，再核对贯穿的线迹；外边不计作内部分隔线。不得因编号不连续而补线，也不得用恢复图的行列数解释原图。看不清交点时写uncertain。',
                  'repair_existing只处理尚未正确对应的原图部件(missing)或没有独立原图依据的现有副本(extra)。缺失且没有对应部件用空part_ids；extra要指定实际part_ids。不另建同一区域的完整副本。位置、尺寸和轮廓由局部复原负责，本阶段不重新拟合。',
                  '恢复的路线若接入了不同的原图对象，不能因已有一条线就算覆盖。其对应原图路线尚未正确恢复时，用missing指定这个现有part_id，交给已有局部修复纠正接入；若正确路线已有另一份恢复且这条错误路线没有独立原图依据，用extra指定错误part_id。只定位身份或连接对应错误，不把同一对象的小幅位置、宽度或颜色误差当作新增实例。',
                  '若缺失成员位于旧候选范围之外，在该repair_existing项返回source_extent，覆盖原图中同一完整对象及其缺失成员。使用当前视图坐标；这只重新定义观察范围，不提供拟合轮廓，不包含相邻独立对象。旧框不能限制新增成员的可见性。原范围足够时可省略。',
                  '候选名称仅帮助查找，不是类别或位置的真值。先按原图可见轮廓与恢复部件建立对应；不要按名称把已存在的部件搬到邻居位置。证据不明确写uncertain。',
                  '按干净原图逐项确认职责范围内的成员归属；候选框或SAM框不代表已恢复。图中连接路线不能因弯曲、分叉或括弧状外观而重新登记为分组符号。',
                  '框线和内部线条可能属于同一个对象。不要仅因候选框或 SAM 框存在就生成外框。',
                  'scope是唯一职责边界；branch_ownership记录同次原图观察的部件归属。已移交的箭头、文字不是shape缺口，不提名补回或另建owner。若原归属确实错误，必须以clean源图和scope说明；不能按图标用途或复合完整性推翻边界。仅登记未对应的职责范围内shape成员。',
                  '同一源部件重复时，保留更贴合原图的一份，在repair_existing用extra指定另一份的part_ids；不要删除整组。无法确定才写possible_duplicates。complete只表示本次未发现待修复内容。'],
              'output_schema': spec}
    if any(o.get('source_registration_conflicts') for o in prompt['object_states']):
        prompt['rules'].append('source_registration_conflicts是已有数值测量发现的原图部件数量冲突。它只提示核对完整性，不提供新增轮廓或数量真值；对照clean逐项定位未对应的真实成员，用已有repair_existing返回缺失内容，保持已确认成员和箭头职责边界。')
    if assets:
        from .assets import IMAGE_OWNERSHIP
        prompt['image_ownership'] = {'rule': IMAGE_OWNERSHIP.replace('Cyan IMAGE footprints', 'The listed IMAGE footprints')}
        prompt['image_assets_in_full_image_pixels'] = [
            {'id': a['id'], 'label': a.get('target', a.get('label', a.get('category', 'image asset'))),
             'bbox_pixels': mapped_box(a['bbox_2d'], item['source_size'], size)} for a in assets]
    if mode == 'norm1000':
        prompt.pop('image_size')
        prompt['local_coordinates'] = {'convention': NORM1000_BOX_CONVENTION}
        candidate_spec['properties']['bbox_2d']['items'].update(minimum=0, maximum=1000)
    else:
        prompt['output_coordinates'] = '当前全图像素坐标；bbox 为包含端点的像素中心，不能超过 image_size 减一。'
    def checker(raw):
        raw=deepcopy(raw);held=raw.pop('held_repairs',[])
        validate(raw, spec)
        if held:raw['held_repairs']=held
        raw=isolate_repair_parts(raw,scene)
        repaired=set()
        for row in raw['repair_existing']:
            oid=row['object_id']
            if oid in repaired:raise ValueError('Combine repairs for the same object')
            repaired.add(oid)
            if 'source_extent' in row and not valid_box(row['source_extent'],model_size(size,mode)):
                raise ValueError('Invalid original-source repair extent')
            if 'source_extent' in row:
                q=to_pixels(row['source_extent'],size,mode)
                b=next(o['bbox_2d'] for o in objects if o['id']==oid)
                if max(q[0],b[0])>=min(q[2],b[2]) or max(q[1],b[1])>=min(q[3],b[3]):
                    raise ValueError('A repair extent must continue the same localized source object')
            available={scene['parts'][k]['id'] for k in scene['objects'][oid]['part_keys']}
            for issue in row['issues']:
                if not set(issue['part_ids'])<=available:raise ValueError('Unknown repair part')
                if issue['kind']!='missing' and not issue['part_ids']:raise ValueError('Existing geometry repair needs part IDs')
        for o in raw['new_objects']:
            if not valid_box(o['bbox_2d'], model_size(size, mode)):
                raise ValueError('Invalid discovery bbox')
        if any(not set(p['object_ids']) <= ids for p in raw['possible_duplicates']):
            raise ValueError('Unknown duplicate candidate ID')
        return raw
    images = [('Clean whole image', workspace/item['input']), ('Existing candidates', folder/'inventory.png'),
              ('Current diagnostic geometry', folder/'fit.png')]
    evidence = getattr(runtime, 'spatial_evidence', None)
    if evidence is not None and hasattr(evidence, 'coverage'):
        represented = []
        for obj in existing:
            parts = [scene['parts'][key] for key in scene['objects'][obj['id']]['part_keys']
                     if scene['parts'][key]['status'] == 'ok']
            boxes = [p['bbox_2d'] for p in parts]
            bounds = ([min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None)
            represented.append({**obj, 'fitted_bbox_2d': bounds, 'fitted_part_boxes': boxes})
        prompt, images = evidence.coverage(prompt, images, item, represented, folder, coordinate_mode=mode)
        checker = evidence.review_validator(checker, folder, represented, isolate_coverage_conflicts=True)
    raw = runtime.call(folder/'calls', 'coverage', SYSTEM, prompt, images, checker, options['coverage_max_tokens'])
    if mode == 'norm1000':
        save(folder/'review_wire.json', raw)
        raw = {**raw, 'new_objects': [{**o, 'bbox_2d': to_pixels(o['bbox_2d'], size, mode)} for o in raw['new_objects']],
            'repair_existing':[{**r,**({'source_extent':to_pixels(r['source_extent'],size,mode)} if 'source_extent' in r else {})}
                               for r in raw['repair_existing']]}
    proposed = []
    for i, o in enumerate(raw['new_objects']):
        sequence=i+1
        identity=digest(o)[:12]
        oid = f'F{round_index+1}_N{sequence:03d}_{identity}'
        while oid in ids:
            sequence+=1
            oid=f'F{round_index+1}_N{sequence:03d}_{identity}'
        ids.add(oid)
        proposed.append({**o, 'id': oid, 'bbox_2d': mapped_box(o['bbox_2d'], size, item['source_size']),
                         'origin': 'fitting_coverage', 'discovery_round': round_index})
    decision = run_filter(proposed, assets,
        {'path': str(workspace/item['original']), 'sha256': item['source_sha256'], 'size': item['source_size']},
        folder/'image_conflict_review', runtime, {**runtime.config, 'iou_threshold': threshold})
    save(folder/'image_dedup.json', decision)
    from .coverage_evidence import bind
    evidence_file=folder/'sam_evidence.json'
    requests=[{**r,**({'source_extent_original':mapped_box(r['source_extent'],size,item['source_size'])}
                      if 'source_extent' in r else {})} for r in raw['repair_existing']]
    repairs=bind(requests,existing,read(evidence_file) if evidence_file.exists() else {},
                 raw.get('sam_decisions',[]))
    record = {'round': round_index, 'review': raw, 'new_candidate_ids': [o['id'] for o in decision['kept']],
              'image_removed_ids': [o['id'] for o in decision['removed']],
              'image_conflict_deferred_ids': decision['deferred_ids'],
              'image_conflict_review_calls': decision['model_request_calls'],
              'duplicate_action': 'repair_explicit_parts', 'new_objects': decision['kept'],
              'repair_existing': repairs}
    save(folder/'record.json', record)
    return record
