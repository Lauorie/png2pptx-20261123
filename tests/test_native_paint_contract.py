from copy import deepcopy
from zipfile import ZipFile

from full_recovery.native_paint import bind


def package(path, fill):
    xml = f'''<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>
      <p:sp><p:nvSpPr><p:cNvPr id="1" name="block"/></p:nvSpPr><p:spPr>{fill}</p:spPr></p:sp>
      </p:spTree></p:cSld></p:sld>'''
    with ZipFile(path, 'w') as archive:
        archive.writestr('ppt/slides/slide1.xml', xml)


def test_native_fill_overrides_preview_metadata_without_mutating_inputs(tmp_path):
    path = tmp_path/'source.pptx'
    package(path, '<a:solidFill><a:srgbClr val="123456"><a:alpha val="50000"/></a:srgbClr></a:solidFill>')
    scene = {'arrows': [{'id': 'block', 'kind': 'block_arrow', 'style': {'color': '#666666',
             'shape_parts': {'stale': {'fill': '#FFFFFF'}}}}]}
    before = deepcopy(scene)
    binary = path.read_bytes()
    result, receipt = bind(scene, path)
    assert result['arrows'][0]['style']['shape_parts'] == {'block': {'fill': '#123456', 'fill_opacity': .5}}
    assert receipt['bound'] and scene == before and path.read_bytes() == binary


def test_missing_or_transformed_fill_does_not_guess_from_preview(tmp_path):
    path = tmp_path/'source.pptx'
    package(path, '<a:solidFill><a:srgbClr val="123456"><a:lumMod val="50000"/></a:srgbClr></a:solidFill>')
    scene = {'arrows': [{'id': 'block', 'kind': 'block_arrow', 'style': {'color': '#666666'}}]}
    result, receipt = bind(scene, path)
    assert receipt['unresolved'] and 'shape_parts' not in result['arrows'][0]['style']


def test_explicit_unfilled_block_retains_its_no_fill_contract(tmp_path):
    path = tmp_path/'source.pptx'
    package(path, '<a:noFill/>')
    result, receipt = bind({'arrows': [{'id': 'block', 'kind': 'block_arrow'}]}, path)
    assert result['arrows'][0]['style']['shape_parts']['block']['fill'] is None

