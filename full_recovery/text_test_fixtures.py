"""Synchronize copied historical tests with the current reference interfaces.

Only frozen tests change. Model prompts, renderer code and text algorithms are
copied verbatim. Legacy tests explicitly select the still-supported legacy mode.
"""
from .prepare import save


def synchronize(backend):
    tests = backend/'tests'
    changes = []
    def edit(name, before, after):
        path = tests/name
        value = path.read_text()
        if before not in value:
            raise ValueError('Text test fixture changed: '+name+' / '+before[:50])
        path.write_text(value.replace(before, after))
        changes.append({'file': name, 'before': before, 'after': after})

    edit('test_single_crop.py', 's.finish(source,out,base,{},',
         "s.finish(source,out,base,{'crop_review_mode':'legacy_single'},")
    edit('test_bullets.py', "assert all(rule in p['rules'] for p in prompts)",
         "assert all(rule in p['rules'] or (p.get('review_rules_version')=='user_r1_r21_20261021' "
         "and 'Restore visible list markers' in ' '.join(p['rules']) "
         "and 'never add a second marker' in ' '.join(p['rules'])) for p in prompts)")
    edit('test_crop_rotation.py',
         "rule=next(r for r in prompt['rules'] if r.startswith('CROP ORIENTATION:'))\n"
         "    for phrase in ('rotation_deg is editable','NEW ABSOLUTE clockwise','old +90 may become -90',\n"
         "                   'logical character order','ORIGINAL CLEAN crop axes','Context IDs remain read-only'):\n"
         "        assert phrase in rule",
         "rule=next(r for r in prompt['rules'] if r.startswith('### R11'))\n"
         "    for phrase in ('Rotation is clockwise','+90 means top-to-bottom','-90 means bottom-to-top',\n"
         "                   'glyph orientation','CLEAN image'):\n"
         "        assert phrase in rule\n"
         "    assert 'Context IDs are read-only neighbors' in ' '.join(prompt['rules'])")
    edit('test_local_run_styles.py', "'MIXED STYLE EXAMPLE'", "'Mixed-Style Example'")
    edit('test_local_run_styles.py', "'not separate text boxes'", "'not as separate text boxes'")
    edit('test_prompt_protocol.py', "        assert all(rule in prompt['rules'] for rule in RECOVERY_SCOPE)",
         "        if prompt.get('review_rules_version')=='user_r1_r21_20261021':\n"
         "            assert len(prompt['rules'])==21\n"
         "            assert 'standalone diagram text' in prompt['rules'][0]\n"
         "            assert 'Exclude: text inside' in prompt['rules'][0]\n"
         "        else:\n"
         "            assert all(rule in prompt['rules'] for rule in RECOVERY_SCOPE)")
    edit('test_prompt_protocol.py', "        assert policy in prompt['rules']",
         "        if prompt.get('review_rules_version')=='user_r1_r21_20261021':\n"
         "            rule=next(r for r in prompt['rules'] if r.startswith('### R9'))\n"
         "            assert all(g in rule for g in ('…','⋮','⋱','editable text','one text unit','dashed borders'))\n"
         "        else:\n"
         "            assert policy in prompt['rules']")
    edit('test_prompt_protocol.py', "'latest valid content and colors'",
         "'latest valid content, colors, and structured run styles'")
    edit('test_prompt_protocol.py', "'omit text when it is absent from the response schema'",
         "'omit them when text is absent from the response schema'")
    edit('test_equation_measurement_cache.py', 'from shape_integration.texts import equations',
         'from shape_integration.texts import equations\nfrom shape_integration import office_render as rendering\nfrom pathlib import Path')
    edit('test_equation_measurement_cache.py', 'equations.subprocess', 'rendering.subprocess')
    edit('test_equation_measurement_cache.py', 'equations.os', 'rendering.os')
    edit('test_equation_measurement_cache.py', '                pdf(target)', '                pdf(self.target)')
    path = tests/'test_equation_measurement_cache.py'
    value = path.read_text()
    before = "    monkeypatch.setattr(rendering.subprocess,'Popen',lambda *a,**k:Process())"
    after = ("    def start(command,*a,**kw):\n"
             "        process=Process();process.target=Path(command[-1]).with_suffix('.pdf')\n"
             "        return process\n"
             "    monkeypatch.setattr(rendering.subprocess,'Popen',start)")
    if value.count(before) != 2:
        raise ValueError('Text renderer cache fixture changed')
    path.write_text(value.replace(before, after, 1))
    changes.append({'file': path.name, 'change': 'Mock fresh local PDF at the current office_render boundary'})

    # This environment's python-pptx exposes a:br as runs, with no Font object
    # unless the break has explicit rPr. Only a:r glyph runs have a mandatory
    # character font. Keep the preceding exact stacked-text/break-order check.
    edit('test_layout_integrity.py',
         "assert {r.font.size for r in shape.text_frame.paragraphs[0].runs}=={shape.text_frame.paragraphs[0].font.size}",
         "glyphs=[r for r in shape.text_frame.paragraphs[0].runs if r._r.tag.endswith('}r')]\n"
         "    assert len(glyphs)==7\n"
         "    assert {r.font.size for r in glyphs}=={shape.text_frame.paragraphs[0].font.size}")

    save(backend/'current_test_fixture_adaptations.json', changes)
