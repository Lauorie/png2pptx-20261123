"""Merge native slide objects and their OPC dependencies without rebuilding them.

Office Math alternate content and its preview image are copied together.
Imported child coordinates and absolute paint sizes use the target EMU frame.
This also avoids renderer-dependent stroke widths in scaled imported groups.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import posixpath
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
MC='http://schemas.openxmlformats.org/markup-compatibility/2006'
SLIDE='ppt/slides/slide1.xml'
RELS='ppt/slides/_rels/slide1.xml.rels'
COMPOSITION_PROTOCOL='local_competitor_branch_choice.147'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_package(path):
    with ZipFile(path) as archive: return {n:archive.read(n) for n in archive.namelist()}


def xml_bytes(node):
    return etree.tostring(node,xml_declaration=True,encoding='UTF-8',standalone=True)


def size(package):
    root=etree.fromstring(package['ppt/presentation.xml'])
    slides=root.find('{'+P+'}sldIdLst')
    if slides is None or len(slides)!=1: raise ValueError('Each stage must provide exactly one slide')
    value=root.find('{'+P+'}sldSz')
    return int(value.get('cx')),int(value.get('cy'))


def normalize_group_frames(tree):
    """Convert child EMUs and absolute paint to target units; retain native paths."""
    audit=[]
    for group in list(tree.iter('{'+P+'}grpSp')):
        transform=group.find('{'+P+'}grpSpPr/{'+A+'}xfrm')
        if transform is None:continue
        values={name:transform.find('{'+A+'}'+name) for name in ('off','ext','chOff','chExt')}
        if any(value is None for value in values.values()):continue
        off,extent,origin,span=(values[k] for k in ('off','ext','chOff','chExt'))
        x,y,cx,cy=(int(off.get('x')),int(off.get('y')),int(extent.get('cx')),int(extent.get('cy')))
        ox,oy,wx,wy=(int(origin.get('x')),int(origin.get('y')),int(span.get('cx')),int(span.get('cy')))
        if wx<=0 or wy<=0 or (x,y,cx,cy)==(ox,oy,wx,wy):continue
        sx,sy=cx/wx,cy/wy
        if abs(sx-sy)>1e-4*max(sx,sy):
            audit.append({'applied':False,'reason':'nonuniform_group_scale'});continue
        tx,ty=x-ox*sx,y-oy*sy
        changed=0
        for child in group:
            if etree.QName(child).localname in ('nvGrpSpPr','grpSpPr','extLst'):continue
            for frame in child.iter('{'+A+'}xfrm'):
                for name,attrs in [('off',('x','y')),('chOff',('x','y')),
                                   ('ext',('cx','cy')),('chExt',('cx','cy'))]:
                    node=frame.find('{'+A+'}'+name)
                    if node is None:continue
                    position=name in ('off','chOff')
                    for attr,factor,shift in zip(attrs,(sx,sy),(tx,ty)):
                        node.set(attr,str(round(int(node.get(attr))*factor+(shift if position else 0))))
                changed+=1
            for node in child.iter():
                local=etree.QName(node).localname;namespace=etree.QName(node).namespace
                fields=(('w',) if local=='ln' else ('sz','spc') if local in ('rPr','defRPr','endParaRPr') else
                        ('lIns','rIns','tIns','bIns') if local=='bodyPr' else
                        ('marL','marR','indent','defTabSz') if local=='pPr' else
                        ('val',) if local in ('spcPts','buSzPts') else
                        ('blurRad','dist','rad','z','extrusionH','contourW') if local in
                            ('outerShdw','innerShdw','softEdge','glow','sp3d') else ())
                if namespace==A:
                    for attr in fields:
                        if node.get(attr) is not None:node.set(attr,str(round(int(node.get(attr))*sx)))
        origin.set('x',str(x));origin.set('y',str(y))
        span.set('cx',str(cx));span.set('cy',str(cy))
        ident=group.find('{'+P+'}nvGrpSpPr/{'+P+'}cNvPr')
        audit.append({'applied':True,'group':ident.get('name','') if ident is not None else '',
                      'scale_xy':[sx,sy],'translation_xy':[tx,ty],'child_transforms':changed})
    return audit


def rel_path(part):
    parent,name=posixpath.split(part)
    return posixpath.join(parent,'_rels',name+'.rels')


def resolve(part,target):
    path=posixpath.normpath(posixpath.join(posixpath.dirname(part),target)) if not target.startswith('/') else target[1:]
    if path.startswith('../') or path=='..': raise ValueError('Relationship escapes its package')
    return path


def matches_name(name, reference):
    return name==reference or name.startswith(reference+'/') or name.startswith(reference+' [')


def filter_objects(tree, references, *, keep=False):
    """Filter native leaves in place, retaining complete parent transforms."""
    metadata = {'nvGrpSpPr', 'grpSpPr', 'extLst'}
    def visit(node, inherited=False):
        tag = etree.QName(node).localname
        if tag in metadata:
            return True
        nv = node.find('./{'+P+'}nvGrpSpPr/{'+P+'}cNvPr') if tag == 'grpSp' else node.find('.//{'+P+'}cNvPr')
        name = nv.get('name', '') if nv is not None else ''
        selected = inherited or any(matches_name(name, ref) for ref in references)
        if tag == 'grpSp':
            for child in list(node):
                if not visit(child, selected): node.remove(child)
            return any(etree.QName(c).localname not in metadata for c in node)
        return selected if keep else not selected
    for node in list(tree):
        if not visit(node): tree.remove(node)


def subset(source, output, references):
    """Isolate existing native objects for evidence, without rebuilding paint."""
    package=read_package(source);root=etree.fromstring(package[SLIDE])
    tree=root.find('{'+P+'}cSld/{'+P+'}spTree')
    filter_objects(tree,set(references),keep=True);package[SLIDE]=xml_bytes(root)
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for name,data in package.items():archive.writestr(name,data)


def suppress_round_caps(tree,selections):
    audit=[]
    for node in tree.iter():
        if etree.QName(node).localname not in ('sp','cxnSp'):continue
        ident=node.find('.//{'+P+'}cNvPr');name=ident.get('name','')if ident is not None else ''
        sides=next((v for k,v in selections.items()if matches_name(name,k)),())
        if not sides:continue
        props=node.find('{'+P+'}spPr');line=props.find('{'+A+'}ln')if props is not None else None
        if line is None:continue
        for side in sides:
            cap=line.find('{'+A+'}'+('headEnd'if side==0 else'tailEnd'))
            if cap is None or cap.get('type')!='oval':continue
            cap.set('type','none');audit.append({'name':name,'side':side,'node_owns_round_cap':True,'shaft_and_other_paint_preserved':True})
    return audit


def suppress_segments(tree, selections):
    """Suppress selected border strokes while retaining the face's closed fill.

Keep its native coordinate frame, transform, IDs and paint. The fill path stays
closed; only the selected redundant border strokes are assigned to arrows.
"""
    audit=[]
    for node in tree.iter('{'+P+'}sp'):
        ident=node.find('.//{'+P+'}cNvPr');name=ident.get('name','') if ident is not None else ''
        indices=next((set(v) for ref,v in selections.items() if matches_name(name,ref)),None)
        if not indices:continue
        props=node.find('{'+P+'}spPr')
        paths=props.find('{'+A+'}custGeom/{'+A+'}pathLst')
        if paths is None or len(paths)!=1:raise ValueError('Segment ownership requires one linear custom contour: '+name)
        path=paths[0];commands=list(path)
        if not commands or commands[0].tag!='{'+A+'}moveTo' or any(etree.QName(c).localname not in ('moveTo','lnTo','close') for c in commands):
            raise ValueError('Segment ownership requires a linear contour: '+name)
        points=[c.find('{'+A+'}pt') for c in commands if c.tag in ('{'+A+'}moveTo','{'+A+'}lnTo')]
        if len(points)>1 and points[-1].attrib==points[0].attrib:points=points[:-1]
        if not indices<=set(range(len(points))):raise ValueError('Invalid native border segment index')
        # No other shape paint or geometry is rebuilt. Reuse every existing
        # point exactly, including the same virtual closing side.
        path.set('stroke','false')
        stroke=etree.SubElement(paths,'{'+A+'}path',**{k:v for k,v in path.attrib.items() if k not in ('fill','stroke')},fill='none',stroke='true')
        previous=None
        for i,start in enumerate(points):
            if i in indices:continue
            if previous!=i-1:
                move=etree.SubElement(stroke,'{'+A+'}moveTo');move.append(deepcopy(start))
            end=etree.SubElement(stroke,'{'+A+'}lnTo');end.append(deepcopy(points[(i+1)%len(points)]))
            previous=i
        audit.append({'name':name,'suppressed_segments':sorted(indices),'closed_fill_path_preserved':True})
    return audit


def suppress_subpaths(tree,selections):
    """Remove only approved disconnected strokes from an unfilled custom path."""
    audit=[]
    for node in tree.iter('{'+P+'}sp'):
        ident=node.find('.//{'+P+'}cNvPr');name=ident.get('name','') if ident is not None else ''
        indices=next((set(v) for ref,v in selections.items() if matches_name(name,ref)),None)
        if not indices:continue
        props=node.find('{'+P+'}spPr');paths=props.find('{'+A+'}custGeom/{'+A+'}pathLst')
        if paths is None or len(paths)!=1 or paths[0].get('fill')!='none':
            raise ValueError('Subpath ownership requires one unfilled custom path: '+name)
        path=paths[0];contours=[]
        for command in path:
            kind=etree.QName(command).localname
            if kind=='moveTo':contours.append([])
            if not contours or kind not in ('moveTo','lnTo','cubicBezTo','quadBezTo','arcTo'):
                raise ValueError('Subpath ownership requires disconnected open strokes: '+name)
            contours[-1].append(command)
        if not indices<=set(range(len(contours))):raise ValueError('Invalid native subpath index')
        # Independent approved removals can together cover every subpath.
        # The scene consumer already omits this exhausted part; remove the
        # exact native leaf too, without touching its owner's other parts.
        if indices == set(range(len(contours))):
            node.getparent().remove(node)
            audit.append({'name':name,'suppressed_subpaths':sorted(indices),
                          'retained_subpaths':0,'whole_native_leaf_removed':True})
            continue
        for i in indices:
            for command in contours[i]:path.remove(command)
        audit.append({'name':name,'suppressed_subpaths':sorted(indices),
                      'retained_subpaths':len(contours)-len(indices),'transform_and_paint_preserved':True})
    return audit


def evidence_path(props):
    preset=props.find('{'+A+'}prstGeom')
    if preset is not None and preset.get('prst')=='rect':
        # Evidence only: isolate a native rectangle's actual edge using its
        # unchanged transform and line paint. The delivered preset stays native.
        geometry=etree.Element('{'+A+'}custGeom')
        for tag in ('avLst','gdLst','ahLst','cxnLst'):etree.SubElement(geometry,'{'+A+'}'+tag)
        etree.SubElement(geometry,'{'+A+'}rect',l='0',t='0',r='r',b='b')
        paths=etree.SubElement(geometry,'{'+A+'}pathLst');poly=etree.SubElement(paths,'{'+A+'}path',w='100000',h='100000')
        for i,(x,y) in enumerate([(0,0),(100000,0),(100000,100000),(0,100000)]):
            command=etree.SubElement(poly,'{'+A+'}moveTo' if i==0 else '{'+A+'}lnTo')
            etree.SubElement(command,'{'+A+'}pt',x=str(x),y=str(y))
        etree.SubElement(poly,'{'+A+'}close');props.replace(preset,geometry)
    path=props.find('{'+A+'}custGeom/{'+A+'}pathLst/{'+A+'}path')
    if path is None:raise ValueError('Partial evidence requires custom geometry or a rectangle')
    return path


def selection_subset(source,output,rows):
    """Isolate the entire decision in place, retaining native order and paint.

    Partial strokes are pruned inside their original objects. Attached heads
    remain complete objects in their original positions in the slide tree.
    """
    refs={ref for row in rows for ref in row['refs']}
    whole={ref for row in rows if 'segment' not in row and 'subpath' not in row for ref in row['refs']}
    whole.update(ref for row in rows for ref in row.get('attachment_refs',[]))
    subset(source,output,refs);package=read_package(output);root=etree.fromstring(package[SLIDE])
    tree=root.find('{'+P+'}cSld/{'+P+'}spTree')
    for node in tree.iter('{'+P+'}sp'):
        name=node.find('.//{'+P+'}cNvPr').get('name','')
        if any(matches_name(name,ref) for ref in whole):continue
        partial=[r for r in rows if matches_name(name,r['refs'][0]) and ('segment' in r or 'subpath' in r)]
        if not partial:continue
        props=node.find('{'+P+'}spPr');path=evidence_path(props)
        mode='segment' if 'segment' in partial[0] else 'subpath'
        if any(mode not in r for r in partial):raise ValueError('Mixed native selection modes: '+name)
        chosen={i for r in partial for i in (r.get('subpaths',[r[mode]]) if mode=='subpath' else [r[mode]])}
        if mode=='subpath':
            count=len(path.findall('{'+A+'}moveTo'))
            if not chosen<=set(range(count)):raise ValueError('Invalid evidence subpath index')
            suppress_subpaths(tree,{name:set(range(count))-chosen})
        else:
            points=[c.find('{'+A+'}pt') for c in path if c.tag in ('{'+A+'}moveTo','{'+A+'}lnTo')]
            if points[-1].attrib==points[0].attrib:points=points[:-1]
            if not chosen<=set(range(len(points))):raise ValueError('Invalid evidence segment index')
            suppress_segments(tree,{name:set(range(len(points)))-chosen})
            # Only an explicitly selected face BORDER omits its contextual fill.
            # Arrowheads always enter through whole/attachment refs above.
            for child in list(props):
                if etree.QName(child).localname in ('solidFill','gradFill','pattFill','blipFill','noFill'):props.remove(child)
            etree.SubElement(props,'{'+A+'}noFill')
    package[SLIDE]=xml_bytes(root)
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for name,data in package.items():archive.writestr(name,data)


def subpath_subset(source,output,references,index):
    selection_subset(source,output,[{'refs':references,'subpath':index}])


def segment_subset(source,output,references,segment_index):
    selection_subset(source,output,[{'refs':references,'segment':segment_index}])


def merge(base,layers,output,*,exclude_names=None,exclude_base_names=(),suppress_base_segments=None,suppress_base_subpaths=None,before_objects=None,between_objects=None,suppress_round_terminals=None):
    """Append [(label, pptx)] above base, retaining every native object."""
    output=Path(output)
    paths=[Path(base),*[Path(p) for _,p in layers]]
    if output.resolve() in {p.resolve() for p in paths}: raise ValueError('Do not overwrite stage outputs')
    identities={str(p):sha(p) for p in paths}
    packages=[read_package(p) for p in paths];target=packages[0];canvas=size(target)
    root=etree.fromstring(target[SLIDE]);tree=root.find('{'+P+'}cSld/{'+P+'}spTree')
    filter_objects(tree,set(exclude_base_names))
    segment_audit=suppress_segments(tree,suppress_base_segments or {})
    subpath_audit=suppress_subpaths(tree,suppress_base_subpaths or {})
    relationships=etree.fromstring(target.get(RELS,('<Relationships xmlns="'+PR+'"/>').encode()))
    types=etree.fromstring(target['[Content_Types].xml'])
    overrides={n.get('PartName') for n in types if etree.QName(n).localname=='Override'}
    defaults={n.get('Extension') for n in types if etree.QName(n).localname=='Default'}
    next_id=max([int(n.get('id')) for n in root.iter('{'+P+'}cNvPr')]+[1])+1
    # Permit an arrow between native children of one compound. Keep each
    # child's complete transform ancestry; never reconstruct its geometry.
    # Groups with aggregate effects cannot be split without changing paint.
    split_audit=[];unsplittable=set()
    references={name for layer in (between_objects or {}).values() for spec in layer.values()
                for side in ('before','after') for name in spec.get(side,[])}
    def names(node):return [p.get('name','') for p in node.iter('{'+P+'}cNvPr')]
    def split_group(node):
        nonlocal next_id
        if node.tag!='{'+P+'}grpSp':return [node]
        props=node.find('{'+P+'}grpSpPr')
        if props is not None and any(n.tag!='{'+A+'}xfrm' for n in props):
            raise ValueError('aggregate_group_effect')
        result=[]
        children=[n for n in node if etree.QName(n).localname not in ('nvGrpSpPr','grpSpPr','extLst')]
        for child in children:
            for leaf in split_group(child):
                wrapper=deepcopy(node)
                for n in list(wrapper):
                    if etree.QName(n).localname not in ('nvGrpSpPr','grpSpPr','extLst'):wrapper.remove(n)
                ident=wrapper.find('{'+P+'}nvGrpSpPr/{'+P+'}cNvPr')
                ident.set('id',str(next_id));next_id+=1
                wrapper.append(deepcopy(leaf));result.append(wrapper)
        return result
    for node in list(tree):
        if node.tag!='{'+P+'}grpSp':continue
        if not any(matches_name(name,ref) for name in names(node)[1:] for ref in references):continue
        try:pieces=split_group(node)
        except ValueError as exc:
            unsplittable.update(ref for ref in references if any(matches_name(name,ref) for name in names(node)[1:]))
            split_audit.append({'group':names(node)[0],'applied':False,'reason':str(exc)});continue
        index=tree.index(node);tree.remove(node)
        for piece in reversed(pieces):tree.insert(index,piece)
        split_audit.append({'group':names(node)[0],'applied':True,'native_children':len(pieces)})
    base_objects=[n for n in tree if n.find('.//{'+P+'}cNvPr') is not None]
    next_rel=1
    used_rels={n.get('Id') for n in relationships};audit=[]
    namespaces=dict(root.nsmap);ignorable=set(root.get('{'+MC+'}Ignorable','').split())
    for (label,path),source in zip(layers,packages[1:]):
        source_size=size(source)
        if abs(canvas[0]/canvas[1]-source_size[0]/source_size[1])>1e-4:
            raise ValueError('Stage canvas aspect ratios differ: '+label)
        source_root=etree.fromstring(source[SLIDE])
        for prefix,uri in source_root.nsmap.items():
            if prefix in namespaces and namespaces[prefix]!=uri:
                raise ValueError('Conflicting XML namespace prefix: '+str(prefix))
            namespaces[prefix]=uri
        ignorable.update(source_root.get('{'+MC+'}Ignorable','').split())
        src_tree=source_root.find('{'+P+'}cSld/{'+P+'}spTree')
        excluded=set((exclude_names or {}).get(label,()))
        src_tree=deepcopy(src_tree);filter_objects(src_tree,excluded)
        cap_audit=suppress_round_caps(src_tree,(suppress_round_terminals or{}).get(label,{}))
        children=[deepcopy(n) for n in src_tree if etree.QName(n).localname not in ('nvGrpSpPr','grpSpPr','extLst')]
        src_rels=etree.fromstring(source.get(RELS,('<Relationships xmlns="'+PR+'"/>').encode()))
        relation_table={r.get('Id'):r for r in src_rels}
        type_root=etree.fromstring(source['[Content_Types].xml'])
        source_types={n.get('PartName','').lstrip('/'):n.get('ContentType') for n in type_root if etree.QName(n).localname=='Override'}
        source_defaults={n.get('Extension'):n.get('ContentType') for n in type_root if etree.QName(n).localname=='Default'}
        part_map={};relation_map={};id_map={}
        def clone_part(part):
            if part in part_map:return part_map[part]
            if part not in source:raise ValueError('Missing related part: '+part)
            dest=posixpath.join('ppt/imported',label,part)
            if dest in target:raise ValueError('Package destination collision: '+dest)
            part_map[part]=dest;target[dest]=source[part]
            if part in source_types and '/'+dest not in overrides:
                etree.SubElement(types,'{'+CT+'}Override',PartName='/'+dest,ContentType=source_types[part]);overrides.add('/'+dest)
            else:
                extension=part.rsplit('.',1)[-1]
                if extension in source_defaults and extension not in defaults:
                    etree.SubElement(types,'{'+CT+'}Default',Extension=extension,ContentType=source_defaults[extension]);defaults.add(extension)
            nested=rel_path(part)
            if nested in source:
                rels=etree.fromstring(source[nested])
                for rel in rels:
                    if rel.get('TargetMode')!='External':
                        child=clone_part(resolve(part,rel.get('Target')))
                        rel.set('Target',posixpath.relpath(child,posixpath.dirname(dest)))
                target[rel_path(dest)]=xml_bytes(rels)
            return dest
        for child in children:
            for node in child.iter():
                if node.tag=='{'+P+'}cNvPr':
                    old=node.get('id')
                    if old not in id_map:id_map[old]=str(next_id);next_id+=1
                    node.set('id',id_map[old])
        for child in children:
            for node in child.iter():
                if node.tag in ('{'+A+'}stCxn','{'+A+'}endCxn') and node.get('id') in id_map:node.set('id',id_map[node.get('id')])
                if node.get('spid') in id_map:node.set('spid',id_map[node.get('spid')])
                for key,old in list(node.attrib.items()):
                    if not key.startswith('{'+R+'}'):continue
                    if old not in relation_map:
                        if old not in relation_table:raise ValueError('Missing slide relationship '+old)
                        while 'rId'+str(next_rel) in used_rels:next_rel+=1
                        new='rId'+str(next_rel);used_rels.add(new);next_rel+=1
                        rel=deepcopy(relation_table[old]);rel.set('Id',new)
                        if rel.get('TargetMode')!='External':
                            dest=clone_part(resolve(SLIDE,rel.get('Target')))
                            rel.set('Target',posixpath.relpath(dest,posixpath.dirname(SLIDE)))
                        relationships.append(rel);relation_map[old]=new
                    node.set(key,relation_map[old])
        group=etree.Element('{'+P+'}grpSp')
        nv=etree.SubElement(group,'{'+P+'}nvGrpSpPr')
        etree.SubElement(nv,'{'+P+'}cNvPr',id=str(next_id),name=label);next_id+=1
        etree.SubElement(nv,'{'+P+'}cNvGrpSpPr');etree.SubElement(nv,'{'+P+'}nvPr')
        transform=etree.SubElement(etree.SubElement(group,'{'+P+'}grpSpPr'),'{'+A+'}xfrm')
        for tag,attrs in [('off',{'x':'0','y':'0'}),('ext',{'cx':str(canvas[0]),'cy':str(canvas[1])}),
                          ('chOff',{'x':'0','y':'0'}),('chExt',{'cx':str(source_size[0]),'cy':str(source_size[1])})]:
            etree.SubElement(transform,'{'+A+'}'+tag,**attrs)
        for child in children:group.append(child)
        tree.append(group)
        placed=[];unplaced=[]
        for child in children:
            ident=child.find('.//{'+P+'}cNvPr')
            name=ident.get('name') if ident is not None else ''
            spec=(between_objects or {}).get(label,{}).get(name)
            owners=set(spec.get('before',[]) if spec is not None else (before_objects or {}).get(label,{}).get(name,()))
            after=set(spec.get('after',[]) if spec is not None else [])
            targets=[n for n in base_objects if any(matches_name(v,o) for v in names(n) for o in owners)]
            behind=[n for n in base_objects if any(matches_name(v,o) for v in names(n) for o in after)]
            missing=[ref for ref in owners|after if not any(matches_name(v,ref) for n in base_objects for v in names(n))]
            if spec is not None and (missing or (owners|after)&unsplittable):
                unplaced.append({'name':name,'reason':'unavailable_native_part_placement',
                                 'missing':missing,'unsplittable':sorted((owners|after)&unsplittable)});continue
            if not targets:continue
            upper=min(tree.index(n) for n in targets)
            lower=max((tree.index(n)+1 for n in behind),default=0)
            if lower>upper:
                unplaced.append({'name':name,'before':sorted(owners),'after':sorted(after),
                                 'reason':'no_slot_preserving_source_visible_parts'});continue
            earlier=deepcopy(group)
            for n in list(earlier):
                if etree.QName(n).localname not in ('nvGrpSpPr','grpSpPr'):earlier.remove(n)
            nv=earlier.find('{'+P+'}nvGrpSpPr/{'+P+'}cNvPr')
            nv.set('id',str(next_id));next_id+=1;nv.set('name',label+' '+ident.get('name'))
            group.remove(child);earlier.append(child)
            tree.insert(upper,earlier)
            placed.append({'name':ident.get('name'),'before_owners':sorted(owners),'after_parts':sorted(after)})
        audit.append({'layer':label,'source':str(path),'source_sha256':identities[str(Path(path))],
                      'native_objects':len(children),'related_parts':len(part_map),'object_ids_remapped':len(id_map),
                      'source_canvas':source_size,'target_canvas':canvas,
                      'ownership_excluded_names':sorted(excluded),
                      'source_node_round_terminal_ownership':cap_audit,
                      'source_occlusion_placements':placed,
                      'source_occlusion_unresolved':unplaced,
                      'native_math':sum(1 for n in group.iter() if etree.QName(n).localname=='oMath')})
    # Requires="a14" is a prefix-valued attribute, so retain source namespace
    # declarations even if that prefix appears only inside alternate content.
    merged=etree.Element(root.tag,nsmap=namespaces)
    merged.attrib.update(root.attrib)
    if ignorable:merged.set('{'+MC+'}Ignorable',' '.join(sorted(ignorable)))
    for child in root:merged.append(child)
    coordinate_audit=normalize_group_frames(merged.find('{'+P+'}cSld/{'+P+'}spTree'))
    target[SLIDE]=xml_bytes(merged);target[RELS]=xml_bytes(relationships);target['[Content_Types].xml']=xml_bytes(types)
    for element in merged.iter('{'+MC+'}Choice'):
        for prefix in element.get('Requires','').split():
            if prefix not in element.nsmap:raise ValueError('Lost alternate-content namespace '+prefix)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.partial')
    with ZipFile(temporary,'w',ZIP_DEFLATED) as archive:
        for name,body in target.items():archive.writestr(name,body)
    with ZipFile(temporary) as archive:
        if archive.testzip():raise ValueError('Corrupt composed package')
    if any(sha(p)!=identities[str(p)] for p in paths):raise ValueError('A stage source changed during composition')
    temporary.replace(output)
    return {'base_sha256':identities[str(Path(base))],'output_sha256':sha(output),'layers':audit,
            'group_coordinate_normalization':coordinate_audit,
            'native_group_splits':split_audit,
            'native_border_segment_ownership':segment_audit,
            'native_subpath_ownership':subpath_audit,
            'stage_sources_unchanged':True,'native_objects_copied_without_geometry_reconstruction':True}
