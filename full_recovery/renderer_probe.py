"""Verify actual native-PPT conversion before starting expensive inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def validate(executable, output, timeout=120):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
    from pptx.util import Inches
    import fitz
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    pptx = output/'render_smoke.pptx'
    prs = Presentation(); prs.slide_width = Inches(10); prs.slide_height = Inches(5.625)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    face = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(1), Inches(1), Inches(3), Inches(1))
    face.text = 'PPT render smoke'
    slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4), Inches(1.5), Inches(8), Inches(3))
    prs.save(pptx)
    with tempfile.TemporaryDirectory(prefix='ppt-render-smoke-') as profile:
        argv = [str(Path(executable).resolve()), '-env:UserInstallation='+Path(profile).as_uri(),
                '--headless', '--convert-to', 'pdf', '--outdir', str(output), str(pptx)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    pdf = pptx.with_suffix('.pdf')
    receipt = {'executable': str(executable), 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr, 'pdf_exists': pdf.is_file(),
               'pptx_sha256': hashlib.sha256(pptx.read_bytes()).hexdigest()}
    if pdf.is_file():
        with fitz.open(pdf) as document:
            receipt['pages'] = len(document)
            receipt['text_present'] = len(document) == 1 and 'PPT render smoke' in document[0].get_text()
        receipt['pdf_sha256'] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    receipt['passed'] = result.returncode == 0 and receipt.get('pages') == 1 and receipt.get('text_present', False)
    (output/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not receipt['passed']:
        raise RuntimeError('Native PPT render smoke failed before inference: '+str(output/'validation.json'))
    return receipt
