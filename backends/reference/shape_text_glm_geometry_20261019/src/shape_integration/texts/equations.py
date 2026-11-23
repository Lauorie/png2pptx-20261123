"""Native Office Math, measured from the same OMML with an explicit LO preview.

The deliverable retains editable a14:m. LibreOffice 7.x cannot render that
PowerPoint extension, so its preview uses transparent images rendered from the
same expression in a temporary Word document, NOT a crop of the source image.
"""
from copy import deepcopy
from difflib import SequenceMatcher
import unicodedata
import io
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

import fitz
from lxml import etree
import numpy as np
from PIL import Image
from ..resource_limits import bounded

from .omml import latex_to_omml, OMML_NS, DRAWINGML_NS, A14_NS, MC_NS, PML_NS

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'m': OMML_NS, 'a': DRAWINGML_NS, 'p': PML_NS, 'a14': A14_NS, 'mc': MC_NS}


def element(ns, tag, **attrs):
    node = etree.Element('{%s}%s' % (ns, tag))
    for key, value in attrs.items():
        node.set('{%s}%s' % (ns, key), str(value))
    return node


def native_tree(row, size=24):
    profile=row.get('native_formula_layout',{})
    return latex_to_omml(row['latex'],font_size_pt=size,color=row['color'],
        numeric_vectors=bool(profile.get('numeric_vectors')),
        vector_gap_em=profile.get('vector_gap_em'),
        numeric_font_family=profile.get('numeric_font_family'))


def word_math(row, black=False, size=24, literal_preview=False):
    tree = native_tree(row,size)
    # StarMath (LibreOffice's DOCX equation importer) treats a leading '='
    # or isolated superscript '*' as an incomplete operator and draws ¿.
    # In this disposable measurement document only, quote their literal glyphs
    # with m:nor. The delivered native PPT equation remains ordinary OMML math.
    for run in tree.findall('.//m:r', NS):
        text = run.find('m:t', NS)
        script=next((a for a in run.iterancestors() if a.tag in ('{'+OMML_NS+'}sup','{'+OMML_NS+'}sub')),None)
        script_text=''.join(script.xpath('.//m:t/text()',namespaces=NS)).strip() if script is not None else ''
        literal_sign=(text is not None and text.text in ('+','-','−')
                      and re.fullmatch(r'\d*[+−-]',script_text) is not None)
        alphabet=run.find('m:rPr/m:scr',NS)
        if text is not None and ((literal_preview and alphabet is None) or text.text in ('=', '*', 'arg') or literal_sign):
            properties = run.find('m:rPr', NS)
            if properties is None:
                properties = element(OMML_NS, 'rPr')
                run.insert(0, properties)
            if properties.find('m:nor', NS) is None:
                properties.append(element(OMML_NS, 'nor'))
    for prop in list(tree.findall('.//a:rPr', NS)):
        replacement = element(W, 'rPr')
        color = prop.find('a:solidFill/a:srgbClr', NS)
        replacement.append(element(W, 'color', val='000000' if black else color.get('val') if color is not None else row['color'][1:]))
        replacement.append(element(W, 'sz', val=round(size * 2)))
        latin=prop.find('a:latin',NS)
        family=latin.get('typeface') if latin is not None else 'Cambria Math'
        replacement.append(element(W, 'rFonts', ascii=family, hAnsi=family))
        if literal_preview:
            # Literal quoting must not turn mathematical variables upright.
            style=prop.getparent().find('m:rPr/m:sty', NS)
            value=style.get('{'+OMML_NS+'}val','i') if style is not None else 'i'
            if value in ('i','bi') and prop.get('i')!='1':replacement.append(element(W,'i'))
            if value in ('b','bi') and prop.get('b')!='1':replacement.append(element(W,'b'))
        for style in ('b','i'):
            if prop.get(style)=='1':replacement.append(element(W,style))
        prop.getparent().replace(prop, replacement)
    return tree


def write_measurement_docx(rows, path, black=False, literal_preview=False):
    document = etree.Element('{%s}document' % W, nsmap={'w': W, 'm': OMML_NS})
    body = etree.SubElement(document, '{%s}body' % W)
    for index, row in enumerate(rows):
        p = etree.SubElement(body, '{%s}p' % W)
        props = etree.SubElement(p, '{%s}pPr' % W)
        if index:
            props.append(element(W, 'pageBreakBefore'))
        props.append(element(W, 'spacing', before=0, after=0))
        math = element(OMML_NS, 'oMathPara')
        math_props = element(OMML_NS, 'oMathParaPr')
        math_props.append(element(OMML_NS, 'jc', val='left'))
        math.append(math_props)
        math.append(word_math(row, black=black, literal_preview=literal_preview))
        p.append(math)
    section = etree.SubElement(body, '{%s}sectPr' % W)
    section.append(element(W, 'pgSz', w=31680, h=7200))
    section.append(element(W, 'pgMar', top=720, bottom=720, left=720, right=720, header=0, footer=0, gutter=0))
    content_types = b'''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'''
    relationships = b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', content_types)
        archive.writestr('_rels/.rels', relationships)
        archive.writestr('word/document.xml', etree.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True))


@bounded('render')
def render_docx(path, folder, timeout):
    from ..office_render import convert_pdf
    folder.mkdir(parents=True, exist_ok=True)
    conversion=convert_pdf(path,folder,timeout,log_name='render.log')
    result=conversion['pdf']
    if not conversion['pdf_published'] or not complete_pdf(path,result):
        raise RuntimeError('OMML measurement render failed, stale or incomplete: '+conversion['log'][-1500:])
    if conversion['timed_out'] or conversion['returncode']:
        (folder/'completed_pdf_recovery.json').write_text(json.dumps({
            'timeout':conversion['timed_out'],'returncode':conversion['returncode'],
            'policy':'complete PDF with matching expected pages; no formula retry'},indent=2))
    if not result.is_file():raise RuntimeError('OMML measurement PDF missing')
    return result


def complete_pdf(document, result):
    """Reject partial/truncated output; validate expected measurement pages."""
    try:
        if not result.is_file() or not result.read_bytes().rstrip().endswith(b'%%EOF'):return False
        with ZipFile(document) as archive:
            tree=etree.fromstring(archive.read('word/document.xml'))
        expected=len(tree.findall('.//m:oMathPara',NS))
        with fitz.open(result) as pdf:
            return expected>0 and not pdf.is_repaired and len(pdf)==expected and all(
                bool(page.get_text().strip()) for page in pdf)
    except (OSError,ValueError,RuntimeError,BadZipFile,etree.XMLSyntaxError):return False


def measure(rows, folder, slots, timeout=180, *, literal_preview=False, failures=None):
    """Return actual equation ink bounds and independently rendered alpha assets."""
    if not rows:
        return {}
    folder.mkdir(parents=True, exist_ok=True)
    for black, name in [(True, 'black')]:
        document = folder / (name + '.docx')
        prior_xml=None
        if document.is_file():
            try:
                with ZipFile(document) as archive:prior_xml=archive.read('word/document.xml')
            except (OSError,ValueError,BadZipFile):pass
        write_measurement_docx(rows, document, black=black, literal_preview=literal_preview)
        with ZipFile(document) as archive:current_xml=archive.read('word/document.xml')
        cached_pdf=folder/name/(name+'.pdf')
        if prior_xml==current_xml and complete_pdf(document,cached_pdf):
            (folder/'render_cache.json').write_text(json.dumps({'policy':'identical measurement XML and complete PDF',
                'xml_sha256':hashlib.sha256(current_xml).hexdigest(),'api_calls':0},indent=2))
        else:
            with slots:render_docx(document,folder/name,timeout)
    output = {}
    with fitz.open(folder / 'black/black.pdf') as black_pdf:
        if len(black_pdf) != len(rows):
            raise ValueError('Formula overflow changed measurement page count; refusing clipped formulas')
        for index, row in enumerate(rows):
            try:
                output[row['id']] = measure_page(black_pdf[index], row, folder)
                if literal_preview:
                    output[row['id']]['method'] = 'native_omml_docx_literal_preview'
            except (ValueError, RuntimeError) as exc:
                if failures is None:
                    raise
                failures[row['id']] = str(exc)
    (folder / 'measurements.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def measure_page(page, row, folder):
    rendered_text = page.get_text()
    if '¿' in rendered_text and '¿' not in row['latex']:
        raise ValueError('Equation renderer inserted an error glyph: ' + row['id'])
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    black = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).astype(float)
    alpha = 1 - black.min(axis=2) / 255
    yy, xx = np.where(alpha > .03)
    if not len(xx):
        raise ValueError('Empty native formula: ' + row['id'])
    if yy.max() >= black.shape[0] - 30 or xx.max() >= black.shape[1] - 30:
        raise ValueError('Clipped equation measurement: ' + row['id'])
    box = [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]
    x0, y0, x1, y1 = box
    a = alpha[y0:y1, x0:x1]
    # LO ignores some OMML leaf colors; transfer the SAME leaf colors to its glyphs.
    rgb, reference_size, color_audit = colored_glyphs(page, row, box)
    rgba = np.dstack((np.clip(rgb, 0, 255), a * 255)).round().astype(np.uint8)
    asset = folder / (row['id'] + '.png')
    Image.fromarray(rgba).save(asset)
    return {'width': (x1 - x0) / 2, 'height': (y1 - y0) / 2,
        'dx': 0., 'dy': 0., 'reference_size_px': reference_size, 'method': 'native_omml_docx',
        'font': {'requested': 'Cambria Math', 'resolved': 'Office Math / local font substitution', 'fallback': True},
        'equation_preview': str(asset.resolve()), 'latex': row['latex'], 'color_audit': color_audit}


def colored_glyphs(page, row, box):
    """Transfer OMML leaf colors to independently rendered equation glyphs."""
    def rgb(value):
        return [int(value.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    expected = []
    tree = native_tree(row)
    for run in tree.findall('.//m:r', NS):
        color = run.find('a:rPr/a:solidFill/a:srgbClr', NS)
        value = color.get('val') if color is not None else row['color']
        for char in unicodedata.normalize('NFKC', ''.join(run.itertext())):
            if not char.isspace():
                expected.append((char, value))
    actual = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                for glyph in span['chars']:
                    for char in unicodedata.normalize('NFKC', glyph['c']):
                        if not char.isspace():
                            actual.append((char, glyph['bbox'], span['size']))
    x0, y0, x1, y1 = map(int, box)
    image = np.empty((y1-y0, x1-x0, 3), dtype=float)
    image[:] = rgb(row['color'])
    matches = SequenceMatcher(None, ''.join(c[0] for c in expected), ''.join(c[0] for c in actual), autojunk=False)
    sizes = []; matched = 0
    for chunk in matches.get_matching_blocks():
        for offset in range(chunk.size):
            char, bounds, size = actual[chunk.b+offset]
            sizes.append(size); matched += 1
            left, top, right, bottom = bounds
            left=max(0, int(np.floor(left*2))-x0); right=min(x1-x0, int(np.ceil(right*2))-x0)
            top=max(0, int(np.floor(top*2))-y0); bottom=min(y1-y0, int(np.ceil(bottom*2))-y0)
            image[top:bottom, left:right] = rgb(expected[chunk.a+offset][1])
    if not sizes:
        raise ValueError('Cannot calibrate native equation font size: '+row['id'])
    # LO's imported math defaults can differ from requested w:sz. Infer the
    # actual base font from ordinary leaf glyphs, excluding stretched radicals.
    reference_size = float(np.percentile(sizes, 90))
    return image, reference_size, {'matched': matched, 'expected': len(expected), 'rendered': len(actual)}


def append_equation(shape, row, scale):
    paragraph = shape.text_frame.paragraphs[0]
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph.alignment = PP_ALIGN.CENTER
    container = element(A14_NS, 'm')
    math_para = element(OMML_NS, 'oMathPara')
    properties = element(OMML_NS, 'oMathParaPr')
    properties.append(element(OMML_NS, 'jc', val='centerGroup'))
    math_para.append(properties)
    math_para.append(native_tree(row,row['font_size_px'] * scale / 12700))
    container.append(math_para)
    paragraph._p.append(container)


def with_fallback(slide, equation, row, scale):
    measurement = row.get('layout', {}).get('measurement', {})
    path = measurement.get('equation_preview')
    if not path or not Path(path).is_file():
        raise ValueError('Native equation needs its measured compatibility asset')
    image = slide.shapes.add_picture(path, equation.left, equation.top, equation.width, equation.height)
    image.name = 'MATH_PREVIEW ' + row['id']
    image.rotation = equation.rotation
    backing = None
    if row.get('background_fill') is not None:
        # Native PowerPoint uses the equation textbox's own fill. The
        # LibreOffice compatibility branch needs the same backing behind its
        # transparent equation preview; do not lose it when replacing Choice.
        from .background import paint
        backing=slide.shapes.add_textbox(equation.left,equation.top,equation.width,equation.height)
        backing.name='MATH_BACKGROUND '+row['id']
        backing.rotation=equation.rotation
        paint(backing,row)
    native = equation._element
    fallback = image._element
    parent = native.getparent()
    position = parent.index(native)
    parent.remove(native)
    parent.remove(fallback)
    alternate = etree.Element('{%s}AlternateContent' % MC_NS, nsmap={'mc': MC_NS, 'a14': A14_NS})
    choice = etree.SubElement(alternate, '{%s}Choice' % MC_NS)
    choice.set('Requires', 'a14')
    choice.append(native)
    legacy=etree.SubElement(alternate, '{%s}Fallback' % MC_NS)
    if backing is not None:
        parent.remove(backing._element)
        legacy.append(backing._element)
    legacy.append(fallback)
    parent.insert(position, alternate)


def compatibility_copy(source, destination):
    """A disposable LO-only copy. NEVER overwrite the editable deliverable."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError('Compatibility copy must not overwrite native equations')
    count = 0
    with ZipFile(source) as incoming, ZipFile(destination, 'w', ZIP_DEFLATED) as outgoing:
        for info in incoming.infolist():
            data = incoming.read(info.filename)
            if info.filename.startswith('ppt/slides/slide') and info.filename.endswith('.xml'):
                root = etree.fromstring(data)
                for alternate in root.xpath('.//mc:AlternateContent[mc:Choice//a14:m and mc:Fallback]', namespaces=NS):
                    parent, position = alternate.getparent(), alternate.getparent().index(alternate)
                    children = list(alternate.find('mc:Fallback', NS))
                    parent.remove(alternate)
                    for index, child in enumerate(children):
                        parent.insert(position + index, deepcopy(child))
                    count += 1
                data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            outgoing.writestr(info, data)
    return count


def render_preview(pptx, output, size, timeout=180):
    from ..styling.export import render_native
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    copy = output / 'libreoffice_compat.pptx'
    count = compatibility_copy(pptx, copy)
    result = render_native(copy if count else pptx, output, size, timeout)
    (output / 'preview_policy.json').write_text(json.dumps({'delivered_pptx': str(pptx),
        'native_equations': count, 'renderer': 'LibreOffice 7.x',
        'equation_preview': 'same-OMML Word render; not a source image crop' if count else 'not needed',
        'native_powerpoint_visual_verified': False}, indent=2))
    return result
