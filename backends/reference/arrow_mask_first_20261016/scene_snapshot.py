"""One labelled, copied inventory for publication and coverage handoffs."""
import copy


def canonical_rows(results):
    rows=[copy.deepcopy(r) for group in results for r in group.get('rows',[])]
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate drawable path IDs at scene handoff')
    labels={key:f'A{i}' for i,key in enumerate(sorted({r['parent_id'] for r in rows}),1)}
    for row in rows:row['label']=labels[row['parent_id']]
    return rows
