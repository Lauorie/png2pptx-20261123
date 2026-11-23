"""Reuse native text glyphs; composition only selects labels and their backing."""
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS = {'p': P, 'a': A}
FILLS = {f'{{{A}}}{name}' for name in
         ('noFill', 'solidFill', 'gradFill', 'blipFill', 'pattFill', 'grpFill')}


def remove_plain_text(source, output, ids):
    """Remove only named plain glyphs for replacement; preserve other objects."""
    if Path(source).resolve()==Path(output).resolve():
        raise ValueError('The source text product must remain read-only')
    with ZipFile(source) as archive:
        package={n:archive.read(n)for n in archive.namelist()}
    slide=etree.fromstring(package['ppt/slides/slide1.xml']);seen=set()
    for node in slide.findall('.//p:sp',NS):
        name=node.find('p:nvSpPr/p:cNvPr',NS)
        ident=name.get('name','')[4:] if name is not None and name.get('name','').startswith('TXT ') else None
        if ident not in ids:
            continue
        if ident in seen or node.find('p:txBody',NS) is None or node.xpath('.//*[local-name()="oMath"]'):
            raise ValueError('Plain-text replacement has ambiguous or formula identity: '+str(ident))
        seen.add(ident);node.getparent().remove(node)
    if seen!=set(ids):
        raise ValueError('Replacement text IDs missing: '+','.join(sorted(set(ids)-seen)))
    package['ppt/slides/slide1.xml']=etree.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,'w',ZIP_DEFLATED)as archive:
        for name,data in package.items():archive.writestr(name,data)


def exact_audit_duplicates(rows):
    """An audit recall must not duplicate an already localized occurrence.

    Match the complete literal content and exact original-source glyph box.
    Typography is allowed to differ: these are two claims on the same source
    occurrence, and an audit recall does not supersede its localized owner.
    Different boxes or a partial transcription remain independent.
    """
    groups = {}
    for row in rows:
        box = row.get('bbox_2d')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        key = (row.get('kind'), row.get('text'), row.get('latex', ''), tuple(box))
        groups.setdefault(key, []).append(row)
    dropped, audit = [], []
    for group in groups.values():
        owners = [r for r in group if r.get('origin') != 'inventory_visual_audit']
        recalls = [r for r in group if r.get('origin') == 'inventory_visual_audit']
        if not owners or not recalls:
            continue
        owner = min(owners, key=lambda r: r['id'])
        for recall in recalls:
            if recall['id'] == owner['id']:
                continue
            dropped.append(recall['id'])
            audit.append({'text_id': recall['id'], 'owner_text_id': owner['id'],
                          'action': 'drop_exact_audit_duplicate',
                          'source_bbox_2d': recall['bbox_2d'],
                          'complete_literal_content_identical': True,
                          'reason': 'Audit recall repeats the same complete localized source occurrence.'})
    return sorted(set(dropped)), audit


def select_native_text(source, output, rows):
    """Preserve paragraphs, metrics, OMML and fallbacks from the supplied PPTX.

    A scene is a composition index, not a portable serialization of every
    version of the text exporter. Never regenerate glyphs with an older one.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('The supplied text product must remain read-only')
    selected = {r['id']: r for r in rows}
    if len(selected) != len(rows):
        raise ValueError('Duplicate text IDs in composition')
    with ZipFile(source) as archive:
        package = {name: archive.read(name) for name in archive.namelist()}
    slide = etree.fromstring(package['ppt/slides/slide1.xml'])
    tree = slide.find('p:cSld/p:spTree', NS)
    seen = set()
    dot_components = {}
    for node in list(tree)[2:]:
        if node.tag == f'{{{P}}}extLst':
            continue
        names = {n.get('name', '') for n in node.findall('.//p:cNvPr', NS)}
        if names and all(name.startswith('label backing ') for name in names):
            tree.remove(node)
            continue
        from .source_ellipsis import native_dot_identity
        dots = [native_dot_identity(name) for name in names]
        if any(dot is not None for dot in dots):
            if not all(dot is not None for dot in dots) or len({dot[0] for dot in dots}) != 1:
                raise ValueError('Text product contains ambiguous punctuation identity')
            ident = dots[0][0]
            components = {dot[1] for dot in dots}
            previous = dot_components.setdefault(ident, set())
            if previous & components or ident in seen:
                raise ValueError('Duplicate native punctuation component: ' + ident)
            previous.update(components)
            if ident not in selected:
                tree.remove(node)
            continue
        primary = {name for name in names if name.startswith(('TXT ', 'EQ '))}
        if len(primary) != 1:
            raise ValueError('Text product contains ambiguous shape identity')
        name = primary.pop()
        ident = name.split(' ', 1)[1]
        if names - {name, 'MATH_PREVIEW ' + ident}:
            raise ValueError('Formula fallback does not match its native label: ' + name)
        if ident in seen or ident in dot_components:
            raise ValueError('Duplicate native text ID: ' + ident)
        seen.add(ident)
        if ident not in selected:
            tree.remove(node)
            continue
        row = selected[ident]
        color = None if row.get('background_box_px') else row.get('background_fill')
        if color and not re.fullmatch(r'#[0-9A-Fa-f]{6}', color):
            raise ValueError('Invalid native label backing color')
        for props in node.findall('.//p:spPr', NS):
            if props.getparent().tag == f'{{{P}}}pic':
                continue
            for child in list(props):
                if child.tag in FILLS:
                    props.remove(child)
            fill = etree.Element(f'{{{A}}}solidFill' if color else f'{{{A}}}noFill')
            if color:
                etree.SubElement(fill, f'{{{A}}}srgbClr', val=color[1:])
            index = 0
            for i, child in enumerate(props):
                if child.tag in {f'{{{A}}}xfrm', f'{{{A}}}prstGeom', f'{{{A}}}custGeom'}:
                    index = i + 1
            props.insert(index, fill)
    if any(components != {0, 1, 2} for components in dot_components.values()):
        raise ValueError('Incomplete native punctuation occurrence')
    seen.update(dot_components)
    if set(selected) - seen:
        raise ValueError('Selected labels absent from text product: ' + ','.join(sorted(set(selected) - seen)))
    package['ppt/slides/slide1.xml'] = etree.tostring(
        slide, xml_declaration=True, encoding='UTF-8', standalone=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in package.items():
            archive.writestr(name, data)


def remove_plain_text(source, output, ids):
    """Remove uniquely named plain labels while retaining every other OPC part."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('The supplied text product must remain read-only')
    ids = set(ids)
    with ZipFile(source) as archive:
        package = {name: archive.read(name) for name in archive.namelist()}
    slide = etree.fromstring(package['ppt/slides/slide1.xml'])
    tree = slide.find('p:cSld/p:spTree', NS)
    matched = {}
    for node in list(tree)[2:]:
        names = [n.get('name', '') for n in node.findall('.//p:cNvPr', NS)]
        wanted = [name[4:] for name in names if name.startswith('TXT ') and name[4:] in ids]
        if not wanted:
            continue
        if len(wanted) != 1 or names != ['TXT ' + wanted[0]] or node.tag != f'{{{P}}}sp':
            raise ValueError('Ambiguous plain label removal')
        if wanted[0] in matched:
            raise ValueError('Duplicate plain label ID: ' + wanted[0])
        matched[wanted[0]] = node
    if ids - matched.keys():
        raise ValueError('Plain labels absent: ' + ','.join(sorted(ids - matched.keys())))
    for node in matched.values():
        tree.remove(node)
    package['ppt/slides/slide1.xml'] = etree.tostring(
        slide, xml_declaration=True, encoding='UTF-8', standalone=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, data in package.items():
            archive.writestr(name, data)
