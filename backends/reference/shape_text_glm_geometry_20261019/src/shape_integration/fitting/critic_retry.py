"""Repair pure metadata errors without repeating the visual assessment."""
import copy

from ..common import read


def known_fields(raw, spec):
    """Projection ONLY for error classification, never an accepted response."""
    if not isinstance(raw, dict):
        return raw
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k in spec['properties']}
    if isinstance(out.get('issues'), list):
        allowed = spec['properties']['issues']['items']['properties']
        out['issues'] = [{k: v for k, v in i.items() if k in allowed} if isinstance(i, dict) else i
                         for i in out['issues']]
    return out


def retry_builder(system, prompt, images, validator):
    def build(folder, error):
        content = (folder/'content.txt').read_text() if (folder/'content.txt').exists() else None
        raw = read(folder/'parsed_response.json') if (folder/'parsed_response.json').exists() else None
        projected = known_fields(raw, prompt['output_schema'])
        # The compact protocol has no notes field. Nonempty extra text may hide
        # a real defect, so keep the images when deciding what to do with it.
        extras = []
        if isinstance(raw, dict):
            extras.extend(v for k, v in raw.items() if k not in prompt['output_schema']['properties'])
            allowed = prompt['output_schema']['properties']['issues']['items']['properties']
            for issue in raw.get('issues', []) if isinstance(raw.get('issues'), list) else []:
                if isinstance(issue, dict):
                    extras.extend(v for k, v in issue.items() if k not in allowed)
        harmless_extras = all(value is None or value == '' or value == [] or value == {} for value in extras)
        format_only = False
        if raw is not None and projected != raw and ('notes' in prompt['output_schema']['properties'] or harmless_extras):
            try:
                validator(projected)
                format_only = True
            except ValueError:
                pass
        if format_only:
            def preserve_findings(value):
                result = validator(value)
                for key in projected:
                    if key == 'notes':
                        if any(note not in result['notes'] for note in projected['notes']):
                            raise ValueError('format repair removed an existing note; preserve prior findings')
                    elif result[key] != projected[key]:
                        raise ValueError('format repair changed '+key+'; a new visual review is required')
                return result
            return {'system': 'Repair the saved JSON format only. Do not perform a new visual assessment. Return exactly one JSON object.',
                    'prompt': {'version': 'shape_integration.critic_format_repair.1',
                               'validation_error': error, 'previous_response': raw,
                               'task': 'Remove empty extra fields only. Preserve the decision and every issue, including '
                                       'kind, part references, evidence and action, exactly. Do not clear defects or add '
                                       'new conclusions. No images are needed for this formatting repair.',
                               'output_schema': prompt['output_schema']},
                    'images': [], 'validator': preserve_findings}
        # Never silently choose one of conflicting JSON objects, invent a missing field,
        # remap a bad ID or clear material uncertainty. Those require the source images.
        actual = copy.deepcopy(prompt)
        actual.update(validation_error=error, previous_response=raw if raw is not None else content,
                      correction='Resolve the stated error against the supplied images. Preserve supported findings. '
                                 'For conflicting decisions or unknown references, actually verify the evidence; '
                                 'do not erase issues just to make accept valid. Return one complete JSON object, no drafts.')
        return {'system': system, 'prompt': actual, 'images': images, 'validator': validator}
    return build
