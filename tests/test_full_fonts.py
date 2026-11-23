"""Portable text symbols must not inherit equation-font metrics."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from full_recovery.fonts import configure


class PortableTextFonts(unittest.TestCase):
    def test_symbol_fallback_preserves_the_primary_text_and_math_families(self):
        if not shutil.which('fc-match'):
            self.skipTest('fontconfig is unavailable')
        resolved = subprocess.check_output(['fc-match', '-f', '%{family[0]}\n%{file}\n', 'Liberation Sans'], text=True).splitlines()
        if len(resolved) < 2 or resolved[0] != 'Liberation Sans':
            self.skipTest('metric-compatible reference font is unavailable')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); bundle = root/'bundle'; portable = root/'renderer'
            (bundle/'fonts').mkdir(parents=True); (portable/'fonts').mkdir(parents=True)
            font = bundle/'fonts/LiberationSans-Regular.ttf'; shutil.copy2(resolved[1], font)
            digest = hashlib.sha256(font.read_bytes()).hexdigest()
            (bundle/'manifest.json').write_text(json.dumps({'files': {'fonts/'+font.name: digest},
                'resolved': {'Arial': {'resolved': 'Liberation Sans', 'sha256': digest}}}))
            (bundle/'metric-aliases.conf').write_text('<fontconfig><alias binding="strong"><family>Arial</family>'
                '<prefer><family>Liberation Sans</family></prefer></alias></fontconfig>')
            actual = configure(bundle, portable)
            self.assertEqual(actual['Arial']['family'], 'Liberation Sans')
            self.assertEqual(actual['text_symbol_fallback']['family'], 'DejaVu Sans')
            self.assertEqual(actual['Cambria Math']['family'], 'DejaVu Math TeX Gyre')
            self.assertNotEqual(actual['text_symbol_fallback']['sha256'], actual['Cambria Math']['sha256'])
