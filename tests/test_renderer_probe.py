import json
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pytest

from full_recovery.renderer_probe import validate


def test_success_exit_without_pdf_fails_before_inference(tmp_path):
    root = tmp_path/'smoke'
    with patch('full_recovery.renderer_probe.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')):
        with pytest.raises(RuntimeError, match='before inference'):
            validate('/unused/soffice', root)
    receipt = json.loads((root/'validation.json').read_text())
    assert receipt['returncode'] == 0 and receipt['pdf_exists'] is False and receipt['passed'] is False


def test_actual_one_page_pdf_with_expected_text_passes(tmp_path):
    def convert(argv, **kwargs):
        with fitz.open() as document:
            page = document.new_page(); page.insert_text((30, 30), 'PPT render smoke')
            document.save(tmp_path/'smoke/render_smoke.pdf')
        return SimpleNamespace(returncode=0, stdout='converted', stderr='')
    with patch('full_recovery.renderer_probe.subprocess.run', side_effect=convert):
        result = validate('/unused/soffice', tmp_path/'smoke')
    assert result['passed'] is True and result['pages'] == 1 and result['pdf_sha256']
