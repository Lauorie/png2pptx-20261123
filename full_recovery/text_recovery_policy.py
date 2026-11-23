"""Source occurrence safeguards frozen into future text backends.

These checks request joint visual recovery and prevent worsening collisions.
They neither delete text nor use source PDFs, glyph scores, or sample names.
"""
from copy import deepcopy
import ast
import hashlib
from pathlib import Path


PROTOCOL = 'bounded_text_occurrence_recovery.3'


def inventory_repair_prompt(prompt, rejected, accepted):
    """A malformed proposal is not evidence that all but one instance exists."""
    result=deepcopy(prompt)
    result.update(task='Repair the rejected inventory proposal against the actual accepted source ledger.',
        rejected_candidate=deepcopy(rejected['candidate']),validation_error=rejected['reason'],
        accepted_source_occurrences=[{k:deepcopy(row[k]) for k in ('text','bbox_2d','kind') if k in row}
                                     for row in accepted])
    result['rules'] += [
        'The accepted_source_occurrences ledger is the complete set actually recovered so far. '
        'Other rejected proposals have NOT been accepted. Never assume a repeated label exists elsewhere in the ledger.',
        'A malformed proposal can aggregate several distinct physical occurrences. Reread the original CLEAN pixels '
        'and return every missing occurrence represented by this proposal, one ordinary baseline per entry, each with its own tight source box. '
        'Repeated strings at different positions are different occurrences; preserve their actual number and positions, never infer symmetry or a regular count.',
        'Return replaces=[] for every recovered occurrence: no text IDs have been assigned in this inventory phase. '
        'Do not reproduce an already accepted occurrence at the same source position. Do not infer coordinates from invalid replaces IDs.',
        'Return regions=[]. If the proposal has no visible missing occurrence, return texts=[] and explain in uncertain. '
        'Never fabricate text or boxes; report unresolved original content explicitly.']
    return result


def overlap(a, b):
    area = lambda q: max(0, q[2]-q[0])*max(0, q[3]-q[1])
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection/max(1e-9, min(area(a), area(b)))


def new_conflicts(rows, updates, threshold=.20):
    """Existing collisions also need a joint source reread, not an exemption."""
    baseline = {r['id']: r for r in rows}
    active = [(r['id'], child) for r in rows for child in updates.get(r['id'], [r])]
    events = []
    for index, (pa, a) in enumerate(active):
        for pb, b in active[index+1:]:
            now = overlap(a['bbox_2d'], b['bbox_2d'])
            if now < threshold:
                continue
            if pa == pb and (a.get('kind') == 'formula' or b.get('kind') == 'formula'):
                continue
            previous = 0. if pa == pb else overlap(baseline[pa]['bbox_2d'], baseline[pb]['bbox_2d'])
            events.append({'parents': list(dict.fromkeys((pa, pb))),
                'ids': [a['id'], b['id']], 'overlap_smaller_fraction': now,
                'baseline_overlap': previous})
    return events


def reconcile(rows, updates, threshold=.20):
    """An already overlapping baseline cannot authorize a worse collision.

    Existing source occurrences remain present. A rollback is still reported
    as unresolved; it does not certify the baseline's content or position.
    """
    baseline = {r['id']: r for r in rows}
    kept = deepcopy(updates)
    warnings = []
    for _ in range(len(updates)+1):
        active = [(r['id'], child) for r in rows for child in kept.get(r['id'], [r])]
        rejected = set()
        evidence = []
        for index, (pa, a) in enumerate(active):
            for pb, b in active[index+1:]:
                now = overlap(a['bbox_2d'], b['bbox_2d'])
                if now < threshold:
                    continue
                if pa == pb and (a.get('kind') == 'formula' or b.get('kind') == 'formula'):
                    continue
                previous = 0. if pa == pb else overlap(baseline[pa]['bbox_2d'], baseline[pb]['bbox_2d'])
                if now-previous < .15:
                    continue
                parents = {p for p in (pa, pb) if p in kept}
                if parents:
                    rejected.update(parents)
                    evidence.append({'ids': [a['id'], b['id']],
                        'overlap_smaller_fraction': now, 'baseline_overlap': previous})
        if not rejected:
            break
        for parent in sorted(rejected):
            kept.pop(parent)
            warnings.append({'stage': 'layout_conflict', 'id': parent,
                'reason': 'Worsened text-box collision; original occurrence retained but remains unverified',
                'conflicts': [e for e in evidence if any(
                    ident == parent or ident.startswith(parent+'.') for ident in e['ids'])]})
    return kept, warnings


def install(backend):
    """Change a never-executed freeze, leaving the reference source untouched."""
    folder = Path(backend)/'src/shape_integration/texts'
    crop = folder/'two_stage_crop.py'
    integrity = folder/'layout_integrity.py'
    inventory = folder/'inventory_recovery.py'
    protocol = folder/'prompt_protocol.py'
    entry=Path(backend)/'run_global_ocr_review.py'
    if not crop.is_file():
        return {'protocol': PROTOCOL, 'installed': False, 'reason': 'backend_has_no_two_stage_crop'}
    paths = (crop, integrity, inventory, protocol, entry)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source = crop.read_text()
    replacements = [
        ("POLICY = 'two_stage_crop_visual_recall_wide_recovery_no_cpu_gate_v5'",
         "POLICY = 'two_stage_crop_bounded_occurrence_recovery_no_cpu_gate_v7'"),
        ('Text supplied in targets identifies the occurrence; exact transcription and attributes are handled in stage 2.',
         'Previous text supplied in targets may be wrong. Use the original local position and neighboring structure to identify the physical occurrence, independently of spelling; exact transcription is handled in stage 2.'),
        ("        # A target outside its first crop cannot be recovered by another prompt\n"
         "        # over the same pixels. Full-source recovery supplies the missing context.\n"
         "        f=frame(source,folder/'input',{**opt,'crop_min_padding_px':max(64,short*2.5),'crop_padding':0,'crop_scale':100},\n"
         "                None if wide else union(group))",
         "        # Expand the original occurrence's neighborhood at local glyph resolution.\n"
         "        # A failed first correction must not rename its stable position anchor.\n"
         "        padding=max(64,short*2.5)*(2 if wide else 1)\n"
         "        f=frame(source,folder/'input',{**opt,'crop_min_padding_px':padding,'crop_padding':0,'crop_scale':100},\n"
         "                union(group))"),
        ('FULL-FIGURE RECOVERY: previous rough boxes may point at blank space or the wrong diagram node. Locate the complete supplied occurrence from CLEAN, using its text and surrounding structure. Do not remain inside the rough box. Never substitute a different occurrence of a repeated label. If identity is unresolved, return ambiguous; do not guess.',
         'EXPANDED-NEIGHBORHOOD RECOVERY: original rough boxes and previous spelling are fallible. Use CLEAN and neighboring structure to locate the same physical occurrence. Search beyond its rough box but never substitute a matching word from another row or node. If the original occurrence is outside this expanded view or identity is unresolved, return ambiguous; do not guess.'),
        ("'scope':'full_figure_recovery' if wide else 'local_crop'",
         "'scope':'expanded_neighborhood_recovery' if wide else 'local_crop'"),
        ("            parts=updates.get(row['id'])\n"
         "            # Parent IDs are stable; a split parent is reread as a complete unit.\n"
         "            recovery_rows.append(parts[0] if parts and len(parts)==1 else row)",
         "            # Keep the initial spatial anchor even when the first update jumped rows.\n"
         "            recovery_rows.append(deepcopy(row))"),
        ("    # Clear legacy fills, including baseline fallback rows; no CPU fill chooser.",
         "    remaining_conflicts=new_conflicts(rows,selected_parts)\n"
         "    errors.extend({'stage':'unresolved_layout_conflict',\n"
         "        'reason':'Final occurrence boxes still overlap; retained source occurrences need review',**event}\n"
         "        for event in remaining_conflicts)\n"
         "    # Clear legacy fills, including baseline fallback rows; no CPU fill chooser.")]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Text source policy changed; inspect the backend before adapting '+crop.name)
        source = source.replace(old, new, 1)
    code = integrity.read_text()
    definitions = [node for node in ast.parse(code).body
                   if isinstance(node, ast.FunctionDef) and node.name in ('new_conflicts', 'reconcile')]
    if {node.name for node in definitions} != {'new_conflicts', 'reconcile'} or len(definitions) != 2:
        raise ValueError('Text integrity interface changed; inspect before adapting')
    lines = code.splitlines(keepends=True)
    for node in sorted(definitions, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = []
    code = ''.join(lines)+'\nfrom .source_occurrence_policy import new_conflicts, reconcile\n'
    for value in (source, code):
        ast.parse(value)
    crop.write_text(source)
    integrity.write_text(code)
    code=inventory.read_text()
    first="        prompt['task']='Repair ONLY one rejected inventory occurrence in the original full-image coordinates.'"
    last="            'and explain this in uncertain. Never fabricate coordinates or characters.']"
    if code.count(first)!=1 or code.count(last)!=1:
        raise ValueError('Inventory recovery source changed; inspect before adapting')
    start=code.index(first);end=code.index(last,start)+len(last)
    code=code[:start]+"        from .source_occurrence_policy import inventory_repair_prompt\n        prompt=inventory_repair_prompt(prompt,rejected,value['texts'])"+code[end:]
    ast.parse(code);inventory.write_text(code)
    code=protocol.read_text()
    anchor="        rules.append(f'Propose at most {max_regions} coherent regions needing enlarged reading, each with bbox_2d and reason. Otherwise regions=[].')"
    if code.count(anchor)!=1:
        raise ValueError('Inventory prompt source changed; inspect before adapting')
    replacement=anchor+"\n        rules.append('This is the initial source inventory: no text IDs exist yet. Return replaces=[] for every entry. '\n                     'Each independently visible ordinary baseline is one occurrence, including every repetition. '\n                     'Never group equal strings or use invented replaces IDs to summarize repeated occurrences.')"
    code=code.replace(anchor,replacement,1);ast.parse(code);protocol.write_text(code)
    code=entry.read_text()
    old="'review_policy':'two_stage_crop_visual_recall_wide_recovery_no_cpu_gate_v5'"
    new="'review_policy':'two_stage_crop_bounded_occurrence_recovery_no_cpu_gate_v7'"
    if code.count(old)!=1:raise ValueError('Inspect the startup text provenance policy before adapting')
    code=code.replace(old,new,1);ast.parse(code);entry.write_text(code)
    policy = folder/'source_occurrence_policy.py'
    policy.write_text(Path(__file__).read_text())
    # The reference test's fake model returned a coordinate 600 source pixels
    # outside its view. Keep its recovery-before-rollback assertion, with the
    # target outside the first view but visible in the expanded neighborhood.
    # A separate test below checks that genuinely unseen targets stay unresolved.
    recovery_test = Path(backend)/'tests/test_inventory_visual_recovery.py'
    fixture_before = None
    if recovery_test.is_file():
        fixture_source = recovery_test.read_text()
        old = "correct = {'A': [100, 100, 160, 120], 'B': [700, 100, 760, 120]}"
        new = "correct = {'A': [550, 100, 610, 120], 'B': [700, 100, 760, 120]}"
        if fixture_source.count(old) != 1:
            raise ValueError('Inspect reference recovery fixture before adapting its view contract')
        fixture_before = hashlib.sha256(recovery_test.read_bytes()).hexdigest()
        fixture_source = fixture_source.replace(old, new, 1)
        anchor = "            f = read(folder.parent/'input/frame.json')\n            for target in prompt['targets']:"
        verified = ("            f = read(folder.parent/'input/frame.json')\n"
                    "            assert (f['box'][0] <= correct['A'][0]) == wide\n"
                    "            for target in prompt['targets']:")
        if fixture_source.count(anchor) != 1:
            raise ValueError('Inspect reference recovery test frame before adapting')
        recovery_test.write_text(fixture_source.replace(anchor, verified, 1))
    tests = Path(backend)/'tests/test_source_occurrence_policy.py'
    tests.write_text(Path(__file__).with_name('text_recovery_policy_tests.py').read_text())
    return {'protocol': PROTOCOL, 'installed': True, 'base_sha256': before,
            'effective_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (*paths, policy, tests)},
            'recovery_test_adaptation': {'path': str(recovery_test.relative_to(backend)),
                'base_sha256': fixture_before,
                'effective_sha256': (hashlib.sha256(recovery_test.read_bytes()).hexdigest()
                                     if recovery_test.is_file() else None),
                'reason': 'Recovery must use pixels inside its actual view; retain all success assertions'},
            'pixel_acceptance_gate': False, 'automatic_text_deletion': False}
