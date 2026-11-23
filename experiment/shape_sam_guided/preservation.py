"""Offline audit of changed geometry and request replay; not an accuracy score."""
import argparse
from collections import Counter
from pathlib import Path

from core import read, save


def geometry(folder, record):
    selected = record.get('selected_revision')
    if not selected:
        return []
    fit = read(folder / selected / 'fit_original.json')
    return [{k: p.get(k) for k in ['kind', 'status', 'commands']} for p in fit['parts']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases, totals = [], Counter()
    for case in sorted((args.candidate / 'cases').iterdir()):
        if not (case / 'forward_complete.json').exists():
            raise ValueError('Candidate case incomplete: ' + case.name)
        before = {p.parent.name: p for p in (args.reference / 'cases' / case.name / 'fitting/objects').glob('*/record.json')}
        after = {p.parent.name: p for p in (case / 'fitting/objects').glob('*/record.json')}
        objects = []
        for oid in sorted(set(before) | set(after)):
            row = {'id': oid}
            if oid not in after:
                row['change'] = 'absent_from_candidate'
            elif oid not in before:
                row['change'] = 'new_in_candidate'
            else:
                old, new = read(before[oid]), read(after[oid])
                same_identity = old['candidate'] == new['candidate']
                row['change'] = ('identity_changed' if not same_identity else
                                 'identical_geometry' if geometry(before[oid].parent, old) == geometry(after[oid].parent, new)
                                 else 'geometry_changed')
            if oid in after:
                folder = after[oid].parent
                evidence = [read(p) for p in folder.glob('revision_*/sam_evidence.json')]
                row['sam_active'] = any(e.get('shown') for e in evidence)
                row['activation_reasons'] = sorted({e['activation_reason'] for e in evidence
                                                    if e.get('shown') and e.get('activation_reason')})
                row['replayed_calls'] = len(list(folder.rglob('replay.json')))
                row['new_model_calls'] = len(list(folder.rglob('response_0.json')))
                totals['sam_active_objects'] += row['sam_active']
                totals['replayed_object_calls'] += row['replayed_calls']
                totals['new_object_calls'] += row['new_model_calls']
                if row['change'] == 'geometry_changed' and not row['sam_active']:
                    totals['changed_without_sam'] += 1
            totals[row['change']] += 1
            objects.append(row)
        cases.append({'id': case.name, 'counts': dict(Counter(o['change'] for o in objects)), 'objects': objects})
    save(args.output, {'reference': str(args.reference), 'candidate': str(args.candidate),
         'totals': dict(totals), 'cases': cases,
         'note': 'Exact kind/status/path comparison, ignoring namespaces. Preservation is not visual accuracy. '
                 'Coverage additions are compared separately; missing valid cached responses can require fresh inference.'})
    print(dict(totals))


if __name__ == '__main__':
    main()
