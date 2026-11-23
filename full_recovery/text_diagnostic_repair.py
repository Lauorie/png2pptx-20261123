"""Route source diagnostics through existing bounded text recovery calls."""
from copy import deepcopy

PROTOCOL = 'bounded_source_text_diagnostic_repair.3'


def formula_structure_conflicts(rows, updates):
    """Compare parsed math structure and glyphs, ignoring layout/colour wrappers.

    Disagreement nominates the existing source reread. Neither reading is
    authoritative, and the diagnostic never rewrites the expression.
    """
    from shape_integration.texts.omml import latex_to_omml, OMML_NS
    from lxml import etree
    def fingerprint(latex):
        root = latex_to_omml(latex)
        def walk(node):
            if etree.QName(node).namespace != OMML_NS:
                return None
            tag = etree.QName(node).localname
            if tag.endswith('Pr') or tag in ('ctrlPr', 'argPr', 'rPr'):
                return None
            children = tuple(value for child in node
                             if (value := walk(child)) is not None)
            if tag == 'r':
                return children[0] if len(children) == 1 else ('run', children)
            if tag == 't':
                return ('glyph', node.text or '')
            # Math text wrappers may pack a word in one run or one run per
            # letter. Join adjacent literal tokens at the same structural level;
            # never cross a fraction, script, radical or matrix boundary.
            normalized = []
            for child in children:
                if child[0] == 'glyph' and normalized and normalized[-1][0] == 'glyph':
                    normalized[-1] = ('glyph', normalized[-1][1] + child[1])
                else:
                    normalized.append(child)
            return (tag, tuple(normalized))
        return walk(root)
    def literal_signature(row):
        if row.get('kind')=='formula':
            root=latex_to_omml(row.get('latex',''))
            tokens=[]
            for node in root.iter():
                if etree.QName(node).localname!='t':continue
                structure=tuple(etree.QName(parent).localname for parent in reversed(list(node.iterancestors()))
                    if etree.QName(parent).localname in ('sub','sup','f','num','den','rad','deg','m','mr','limLow','limUpp','bar','acc'))
                tokens.extend((structure,g) for g in node.text or '' if not g.isspace())
            return tokens
        runs=row.get('runs') or [{'text':row.get('text',''),'script':'normal'}]
        return [(('sub',) if r.get('script')=='sub' else ('sup',) if r.get('script')=='super' else (),g)
                for r in runs for g in r.get('text','') if not g.isspace()]
    conflicts = []
    for old in rows:
        children = updates.get(old['id'], [])
        if len(children)!=1:continue
        new=children[0]
        if not any(r.get('kind')=='formula' or any(run.get('script','normal')!='normal'
                   for run in r.get('runs',[])) for r in (old,new)):continue
        try:
            if old.get('kind')==new.get('kind')=='formula':
                different=fingerprint(old.get('latex',''))!=fingerprint(new.get('latex',''))
            else:different=literal_signature(old)!=literal_signature(new)
        except (ValueError,TypeError):continue
        if different:
            conflicts.append({'id':old['id'],'stage':'formula_structure_disagreement',
                'initial_latex':old.get('latex',''),'attribute_latex':new.get('latex',''),
                'reason':'Independent source readings disagree on mathematical glyphs or scripts; reread the same occurrence'})
    return conflicts


def repeated_style_diagnostics(rows):
    """Repeated source labels nominate inconsistent weight, never copy style."""
    warnings=[]
    for row in rows:
        if row.get('kind','text')!='text' or row.get('visual_line_count',1)!=1:
            continue
        text=row.get('text','')
        if not text.strip() or len(text)>64:continue
        box=row['bbox_2d'];height=box[3]-box[1]
        if height<=0:continue
        peers=[other for other in rows if other['id']!=row['id']
               and other.get('kind','text')=='text' and other.get('text')==text
               and other.get('visual_line_count',1)==1
               and abs((other.get('rotation_deg',0)-row.get('rotation_deg',0)+180)%360-180)<1
               and .7 <= (other['bbox_2d'][3]-other['bbox_2d'][1])/height <= 1.4]
        opposite=[other['id'] for other in peers if other.get('bold',False)!=row.get('bold',False)]
        same=[other['id'] for other in peers if other.get('bold',False)==row.get('bold',False)]
        if len(opposite)>=3 and len(opposite)>2*(len(same)+1):
            warnings.append({'id':row['id'],'stage':'repeated_label_weight',
                'reason':'Similar source-sized repeated labels have inconsistent predicted weight; reread this occurrence',
                'peer_ids':opposite})
    return warnings


def ask_inventory(runtime, folder, prompt, images, validator, opt):
    from shape_integration.texts import model
    from shape_integration.common import save
    result = model.ask(runtime, folder, prompt, images, validator, opt)
    rejected = result.get('rejected_texts', [])
    if not rejected:
        return result
    retry = deepcopy(prompt)
    retry['task'] = 'Reread only the rejected inventory observations from the same CLEAN region.'
    retry['rejected_observations'] = deepcopy(rejected)
    retry['accepted_corrections_read_only'] = deepcopy(result['texts'])
    retry['rules'] += [
        'Accepted corrections are already retained. Do not return them again. '
        'Use the rejected source positions as fallible identity anchors; reread their complete '
        'visible occurrences and return ordinary valid records using the existing schema. '
        'Do not infer characters from the rejected spelling or move a repeated label to a neighbor.',
        'Only return corrections/additions represented by the rejected observations. '
        'Return duplicate_ids=[]; uncertain observations remain explicit rather than fabricated.']
    repaired = None
    try:
        repaired = model.ask(runtime, folder/'rejected_recovery', retry, images,
                             validator, {**opt, 'protocol_retries': 0})
        # Revalidate the combined response for cross-entry replacement/ID
        # conflicts. Valid original edits stay intact if recovery conflicts.
        combined = validator({**result, 'texts': result['texts'] + repaired['texts']})
        combined['rejected_texts'] = repaired.get('rejected_texts', [])
        combined['uncertain'] = list(dict.fromkeys(result.get('uncertain', []) + repaired.get('uncertain', [])))
        result = combined
        outcome = {'state': 'recovered' if not result['rejected_texts'] and not result['uncertain'] else 'needs_review',
                   'remaining': result['rejected_texts'], 'uncertain': result['uncertain']}
    except Exception as exc:
        outcome = {'state': 'needs_review', 'error': str(exc), 'remaining': rejected}
    result['diagnostic_recovery'] = {'protocol': PROTOCOL, 'requested': len(rejected),
                                     'bounded_recovery_calls': 1, **outcome}
    save(folder/'diagnostic_recovery.json', result['diagnostic_recovery'])
    return result


def adapt_inventory(source):
    before = '            reply = model.ask(runtime, root, prompt,'
    after = ('            from full_recovery.text_diagnostic_repair import ask_inventory\n'
             '            reply = ask_inventory(runtime, root, prompt,')
    if source.count(before) != 1:
        raise ValueError('Inventory audit recovery call changed')
    source = source.replace(before, after)
    before = "            before = len(result['texts'])"
    after = ("            if reply.get('rejected_texts'):\n"
             "                result.setdefault('regions_needing_review', []).append({\n"
             "                    'stage': 'inventory_rejected_recovery', 'region': index,\n"
             "                    'source_box': f['box'], 'rejected': reply['rejected_texts'],\n"
             "                    'repair_status': 'bounded_reread_exhausted'})\n"
             "            for note in reply.get('uncertain', []):\n"
             "                result.setdefault('regions_needing_review', []).append({\n"
             "                    'stage': 'inventory_source_uncertainty', 'region': index,\n"
             "                    'source_box': f['box'], 'reason': note,\n"
             "                    'repair_channel': 'existing_complete_occurrence_crop_review'})\n" + before)
    if source.count(before) != 1:
        raise ValueError('Inventory audit application changed')
    return source.replace(before, after)


def repair_typography(warnings, rows, selected, recovery_ids, job, fit, reconcile):
    """A flagged size is a source-reread target, never a peer-style vote.

    Reuse the existing geometry/attribute job at the stable parent anchor.
    Each parent receives at most one recovery, including earlier failures.
    """
    by_child = {r['id']: old['id'] for old in rows
                for r in selected.get(old['id'], [old])}
    requested = {by_child[e['id']] for e in warnings if e.get('id') in by_child}
    pending = requested - set(recovery_ids)
    result = deepcopy(selected)
    audit = []
    for index, old in enumerate(rows):
        parent = old['id']
        if parent not in requested:
            continue
        receipt = {'id': parent, 'protocol': PROTOCOL}
        if parent not in pending:
            audit.append({**receipt, 'state': 'needs_review',
                          'reason': 'Existing bounded source reread already used'})
            continue
        try:
            _, parts, _, completed, errors = job(1000 + index, [deepcopy(old)], True)
            if parent not in parts or parent not in completed or not parts[parent]:
                raise ValueError('Diagnostic reread did not confirm the complete source occurrence')
            measured, measurement_errors = fit(parts[parent], parent)
            if {r['id'] for r in measured} != {r['id'] for r in parts[parent]}:
                raise ValueError('Diagnostic text measurement incomplete')
            proposed = {**result, parent: measured}
            proposed, conflicts = reconcile(rows, proposed)
            if parent not in proposed or conflicts:
                raise ValueError('Diagnostic repair creates a worsened layout conflict')
            result = proposed
            audit.append({**receipt, 'state': 'source_reread_applied',
                          'warnings': errors + measurement_errors})
        except Exception as exc:
            audit.append({**receipt, 'state': 'needs_review', 'reason': str(exc)})
    return result, audit


def adapt(source):
    before="        prompt=attribute_prompt(f['size'],units,context);save(folder/'attributes_prompt.json',prompt)"
    after=(before+"\n        from full_recovery.text_reading_views import images as reading_images\n"
           "        source_details=reading_images(units,folder/'input/clean.png',folder/'reading_details')")
    if source.count(before)!=1:raise ValueError('Attribute source-view anchor changed')
    source=source.replace(before,after)
    for before in [
        "[('CLEAN source crop.',folder/'input/clean.png'),('FIXED stage-1 glyph boxes and IDs; do not relocate.',folder/'input/locked_boxes.png')]",
        "[('CLEAN source crop.',folder/'input/clean.png'),('FIXED stage-1 glyph boxes and IDs.',folder/'input/locked_boxes.png')]"]:
        if source.count(before)!=1:raise ValueError('Attribute image contract changed')
        source=source.replace(before,before+'+source_details')
    before="    recovery_ids=unresolved|{p for e in conflicts_before_recovery for p in e['parents']}"
    after=("    from full_recovery.text_diagnostic_repair import formula_structure_conflicts\n"
           "    structure_conflicts=formula_structure_conflicts(rows,updates)\n"
           "    save(review_root/'formula_structure_conflicts.json',structure_conflicts)\n"
           +before+"|{e['id'] for e in structure_conflicts}")
    if source.count(before)!=1:raise ValueError('Existing text recovery selection changed')
    source=source.replace(before,after)
    before = "    # Clear legacy fills, including baseline fallback rows; no CPU fill chooser."
    after = ("    from .typography_audit import audit as audit_typography\n"
             "    from full_recovery.text_diagnostic_repair import repeated_style_diagnostics\n"
             "    typography_warnings = audit_typography(chosen) + repeated_style_diagnostics(chosen)\n"
             "    from full_recovery.text_diagnostic_repair import repair_typography\n"
             "    def fit_diagnostic(parts, parent):\n"
             "        return layout.fit_all(parts, output/'diagnostic_typography'/parent,\n"
             "                              slots, fit_opt, source=None)\n"
             "    selected_parts, diagnostic_repairs = repair_typography(\n"
             "        typography_warnings, rows, selected_parts, recovery_ids, job,\n"
             "        fit_diagnostic, reconcile)\n"
             "    if diagnostic_repairs:\n"
             "        chosen = [child for old in rows for child in selected_parts.get(old['id'], [deepcopy(old)])]\n"
             "        accepted = list(selected_parts)\n"
             "        completed.extend(r['id'] for r in diagnostic_repairs if r['state'] == 'source_reread_applied')\n"
             "    save(output/'diagnostic_repair.json', {'repairs': diagnostic_repairs,\n"
             "        'eligible': 'source-anchored repeated-label typography diagnostics',\n"
             "        'extra_critic_stages': 0, 'maximum_recovery_calls_per_parent': 1})\n" + before)
    if source.count(before) != 1:
        raise ValueError('Final text fill/diagnostic anchor changed')
    source = source.replace(before, after)
    before="    typography_warnings=audit_typography(chosen)"
    after="    typography_warnings=audit_typography(chosen)+repeated_style_diagnostics(chosen)"
    if source.count(before)!=1:raise ValueError('Final text diagnostic audit changed')
    source=source.replace(before,after)
    # Preserve recognition diagnostics instead of certifying missing inventory
    # as clean merely because all accepted IDs were localized.
    before = "    errors.extend(typography_warnings)"
    after = (before + "\n"
             "    errors.extend({'stage': 'diagnostic_recovery', **r} for r in diagnostic_repairs\n"
             "                  if r['state'] == 'needs_review')")
    if source.count(before) != 1:
        raise ValueError('Final text warnings anchor changed')
    return source.replace(before, after)
