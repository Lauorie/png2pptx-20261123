"""Render the two native line branches once, with identical source-led nodes."""
from pathlib import Path
from .compose import subset,selection_subset,merge
from .handoff import json_part_id,source_occlusion


def render(scene, styles, rows, shape_pptx, arrow_pptx, source, folder, renderer):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    keys={k for row in rows for k in row.get('endpoint_nodes',[]) if k in scene['parts']}
    keys=[k for k in scene['parts'] if k in keys and scene['parts'][k]['kind'] in ('ellipse','rectangle','rounded_rectangle')]
    refs=[scene['parts'][k]['object_id']+'/'+json_part_id(k) for k in keys]
    nodes=folder/'nodes.pptx';subset(shape_pptx,nodes,refs)
    result={}
    for branch,original in [('shape',shape_pptx),('arrows',arrow_pptx)]:
        selected=[r for r in rows if r['layer']==branch]
        if not selected:continue
        branch_root=folder/branch;branch_root.mkdir(exist_ok=True)
        selected_ppt=branch_root/'lines.pptx';selection_subset(original,selected_ppt,selected)
        target=branch_root/'with_nodes.pptx'
        if branch=='shape':
            # Subsetting the original package preserves the line/node paint
            # order, including patterns above their own face.
            selected_refs={ref for row in selected for ref in row['refs']}
            subset(shape_pptx,target,[*selected_refs,*refs])
        else:
            raw=[{'id':row['key'],'start':row['points'][0].tolist(),
                  'via':row['points'][1:-1].tolist(),'end':row['points'][-1].tolist(),
                  'heads':row.get('heads',['none','none']),'style':row['style']} for row in selected]
            placements,_=source_occlusion({'parts':{k:scene['parts'][k] for k in keys}},styles,
                                          raw,source,set(),part_placements=True)
            if not placements and not any(styles.get(k,{}).get('fill') for k in keys):
                merge(selected_ppt,[('same_nodes',nodes)],target)
            else:
                merge(nodes,[('arrows',selected_ppt)],target,between_objects={'arrows':placements})
        result[branch]=renderer(target,branch_root/'render',scene['source_size'],180)
    return result
