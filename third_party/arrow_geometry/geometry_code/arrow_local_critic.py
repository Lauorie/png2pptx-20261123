"""One local render -> critic/refine -> render cycle with an atomic patch.

No SAM, pixel sampler, geometry veto, or automatic second review. An expansion
request is an explicit pending task, not a pass or an empty reconstruction.
"""
import copy
from pathlib import Path
from PIL import Image
import arrow_path_protocol as paths
import protocol
from transport.common import read_json as read, write_json as write, sha256


def render(document, clean, folder, export_native=False):
    from lib import viz, native
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=paths.rows(document)
    viz.render(clean, rows, folder)
    write(folder/'document.json',document);write(folder/'rows.json',rows)
    if export_native:native.write(document['size'],rows,[r['style'] for r in rows],folder/'arrows.pptx')
    return rows


def prompt(document,writable,context_size=None,crop_in_context=None):
    return dict(task='Review these rendered local arrow/line targets against CLEAN. Correct visible errors with one minimal patch, or pass.',
        coordinate_frame=dict(name='CLEAN',size=document['size'],context_size=context_size,crop_in_context=crop_in_context),
        writable_targets=list(writable),current=document,head_types=protocol.HEADS,
        instructions=[
            'CLEAN is the original source. RENDER is the actual reconstruction on white. OVERLAY shows current path locations with ID labels; its colors are annotations, not source styles. CONTEXT, if supplied, only locates the crop. All new point coordinates and widths use full-resolution CLEAN pixels.',
            'Check the supplied targets for missing or invented portions, endpoint/corner/curve position errors, incorrect connections, and visible head/color/width/dash mistakes. Preserve correct paths. Do not redesign the diagram, explain its architecture, beautify corners, or rebuild unrelated neighbors.',
            'A target may contain many paths; paths within a target need not form one continuous chain. A stroke path has ordered parts: line has exactly two endpoints; curve has >=3 ordered on-centerline points, interpolated separately. Consecutive parts within a path meet at identical coordinates. Use separate paths for branches. Shared network names and shared-node IDs are not required. Crossings do not imply connections.',
            'Use only visible source evidence. A sharp corner stays line+line; curve is for a visibly curved centerline. An ordinary arrow is a stroke plus endpoint heads. A filled block arrow uses kind=block_arrow with a closed contour (>=3 vertices), no parts and heads none/none. Exclude frame/container boundaries, text/operator strokes, chart axes, and headless dense wiring internal to neural-network icons. Standalone connectors and auxiliary/dimension lines remain eligible.',
            'Estimate model-authored style from actual source ink. width_px is the number of CLEAN pixels across the shaft, not CONTEXT pixels or PowerPoint points. line_style is solid/dashed/dotted. heads=[first,last] follows path point order; internal joints are not heads. Use the head definitions. Do not infer a width from background crossed by an incorrect geometric path.',
            'Return pass when the targets already match sufficiently. Otherwise return only necessary add/replace/delete/update_style edits. replace/add value is a complete path: kind,parts or contour,style. style has color,width_px,line_style,heads. update_style contains only changed style fields and cannot alter geometry. Do not copy untouched paths. A split is delete plus new paths; use new path IDs.',
            'If an endpoint or continuation cannot be judged in this crop, use need_context with a CONTEXT bbox and no edits. If genuinely unresolved, use unresolved with no edits. Short reasons only; return JSON.'],
        output_schema=dict(status='pass|patch|need_context|unresolved',
            issues='optional [{path:"target_id/path_id",type:geometry|style|missing|extra|duplicate,reason:short}]',
            edits='[{op:add|replace|delete|update_style,target:target_id,path:path_id,value:complete_path_for_add_or_replace,style:changed_fields_for_update_style}] or []',
            context_bbox='[x0,y0,x1,y1] in CONTEXT pixels only for need_context; otherwise null'))


def refine(document,clean,folder,model,context=None,crop_in_context=None,writable_targets=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    before=paths.validate(document);revision=paths.revision(before)
    if list(Image.open(clean).size)!=before['size']:raise ValueError('CLEAN size differs from declared coordinate frame')
    writable=[t['id'] for t in before['targets']] if writable_targets is None else list(writable_targets)
    if not set(writable)<={t['id'] for t in before['targets']}:raise ValueError('Unknown writable targets')
    context_size=list(Image.open(context).size) if context else None
    render(before,clean,folder/'before',export_native=True)
    images=[('CLEAN: original local source, coordinate frame',Path(clean)),
            ('RENDER: actual geometry and style on white',folder/'before/styled.png'),
            ('OVERLAY: colored path locations and IDs on original',folder/'before/overlap.png')]
    if context:images.append(('CONTEXT: whole figure, crop locator',Path(context)))
    p=prompt(before,writable,context_size,crop_in_context)
    write(folder/'input_manifest.json',dict(base_revision=revision,writable_targets=writable,
        images=[dict(label=l,path=str(x.resolve()),sha256=sha256(x)) for l,x in images]))
    write(folder/'prompt.json',p);write(folder/'before_document.json',before)
    def validate(answer):
        paths.apply_patch(before,answer,revision,writable,context_size)
        return answer
    try:
        answer=model.ask(folder/'request',p,images,validate)
        after=paths.apply_patch(before,answer,revision,writable,context_size)
        state=dict(state='completed' if answer['status'] in ('pass','patch') else 'pending',decision=answer['status'],
                   edits=len(answer.get('edits',[])),base_revision=revision,after_revision=paths.revision(after))
        write(folder/'answer.json',answer)
        if answer['status']=='need_context':
            write(folder/'pending_context.json',dict(targets=writable,context_bbox=answer['context_bbox'],base_revision=revision,
                original_document=str((folder/'before_document.json').resolve()),reason='Explicit additional context required; retained current geometry'))
    except Exception as exc:
        after=copy.deepcopy(before)
        state=dict(state='failed',error=str(exc),base_revision=revision,after_revision=revision,retained_previous=True)
        write(folder/'error.json',dict(error=str(exc),retained_previous=True))
    render(after,clean,folder/'after',export_native=True)
    write(folder/'after_document.json',after);write(folder/'state.json',state)
    return after,state
