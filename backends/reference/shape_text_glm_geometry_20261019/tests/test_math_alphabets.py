from shape_integration.texts.omml import latex_to_omml,OMML_NS
from shape_integration.texts.equations import NS,word_math


def test_mathcal_multiletter_alphabet_stays_script_in_native_omml():
    tree=latex_to_omml(r'w \sim \mathcal{LN}')
    runs=tree.findall('.//m:r',NS)
    for letter in ('L','N'):
        run=next(r for r in runs if r.find('m:t',NS).text==letter)
        assert run.find('m:rPr/m:scr',NS).get('{'+OMML_NS+'}val')=='script'
        assert run.find('m:rPr/m:sty',NS).get('{'+OMML_NS+'}val')=='p'
    variable=next(r for r in runs if r.find('m:t',NS).text=='w')
    assert variable.find('m:rPr/m:scr',NS) is None
    text=''.join(tree.xpath('.//m:t/text()',namespaces=NS))
    assert '∼' in text and '~' not in text


def test_literal_preview_must_not_disable_script_alphabet():
    row={'id':'label','latex':r'w \sim \mathcal{LN}','color':'#123456'}
    tree=word_math(row,literal_preview=True)
    for run in tree.xpath('.//m:r[m:rPr/m:scr]',namespaces=NS):
        assert run.find('m:rPr/m:nor',NS) is None


def test_mathcal_style_does_not_leak_to_neighboring_letters_or_text():
    tree=latex_to_omml(r'\text{similar }w\sim\mathcal{LN}+x')
    for run in tree.findall('.//m:r',NS):
        text=run.find('m:t',NS).text
        if text in ('w','x') or 'similar' in text:
            assert run.find('m:rPr/m:scr',NS) is None
    assert tree.find('.//m:sSub',NS) is None


def test_fix_does_not_change_existing_E_label_representation():
    tree=latex_to_omml(r'\mathrm{E}_{\mathrm{ext}}')
    assert tree.find('.//m:scr',NS) is None
    assert tree.find('.//m:sSub',NS) is not None


def test_delivered_ppt_keeps_editable_script_letters_and_relation(tmp_path):
    from zipfile import ZipFile
    from lxml import etree
    from PIL import Image
    from shape_integration.texts.export import write
    asset=tmp_path/'measured_formula.png'
    Image.new('RGBA',(180,30),(0,0,0,255)).save(asset)
    row={'id':'label','text':'w~LN','kind':'formula','latex':r'w\sim\mathcal{LN}',
         'runs':[],'font':'Times New Roman','bold':False,'italic':True,
         'underline':False,'color':'#000000','rotation_deg':0,
         'font_size_px':20,'resolved_font':'Office Math','frame_px':[10,10,180,30],
         'layout':{'measurement':{'equation_preview':str(asset)}}}
    target=tmp_path/'math.pptx';write([row],[240,80],target)
    with ZipFile(target) as archive:tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    native=tree.find('.//mc:Choice//a14:m',NS)
    assert native is not None
    script=native.xpath('.//m:r[m:rPr/m:scr[@m:val="script"]]/m:t/text()',namespaces=NS)
    assert script==['L','N']
    assert '∼' in ''.join(native.xpath('.//m:t/text()',namespaces=NS))
