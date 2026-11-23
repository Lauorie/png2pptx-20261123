"""Focused source reading at already located text boxes."""
from copy import deepcopy

PROTOCOL = 'locked_source_unit_attributes.5'


def source_attributes(prompt):
    if prompt.get('stage') != '2_attributes_at_locked_boxes':
        return prompt
    result = deepcopy(prompt)
    result['protocol'] += '_source_units_v2'
    result['targets'] = [{k: v for k, v in target.items()
                          if k in ('id', 'locked_bbox_2d')}
                         for target in result['targets']]
    result['rules'] = [
        'CLEAN contains the source glyphs. The fixed boxes identify the target occurrences; '
        'IDs and read-only neighbors provide context. Return each target ID once, with one '
        'attribute record for that locked unit. Geometry and font size are handled separately.',
        'A locked box identifies an occurrence, not a character clipping window. Read the entire '
        'same source label, including leading/trailing letters or digits slightly outside the box. '
        'Do not truncate a word or number to fit the box, and do not take characters from an '
        'independent neighboring label. Font and layout must adapt to the full transcription.',
        'Transcribe the complete visible characters, repetitions, spaces and punctuation in '
        'ordered runs. Use text for ordinary labels and baseline numbers; use formula and '
        'complete LaTeX for mathematical structure, including its subscripts and superscripts.',
        'Ordinary text runs contain visible Unicode characters, never literal LaTeX commands '
        'such as \\alpha. Render a standalone Greek letter as its glyph in text, or use a complete '
        'formula when there is mathematical structure. Preserve case from CLEAN.',
        'Match the visible font from the allowed fonts, foreground color and run styles. '
        'Use explicit bold, italic and underline values on each run. Determine serif, sans '
        'serif or monospaced appearance from the glyphs.',
        'rotation_deg is the absolute clockwise angle of the text baseline in CLEAN. '
        'Measure the actual slant, including shallow angles. Judge each target\'s glyph orientation '
        'independently of nearby words, arrows or borders.',
        'Use single_line for one baseline, multiline with the actual visual_line_count for '
        'parallel baselines INSIDE this locked unit, never lines of a neighboring label or enclosing '
        'container. Stage 1 separates ordinary baselines. Use stacked_vertical with rotation_deg=0 '
        'only when the individual glyphs are upright in CLEAN; a sideways complete word stays '
        'single_line with its baseline rotation. A complete formula remains one expression.',
        'Standalone ellipses remain one complete text unit using the corresponding glyph. '
        'Preserve visible list markers with their text. Image text is excluded only when '
        'it belongs to a separately embedded image asset.',
        'For formulas, LaTeX defines mathematical structure and appearance; use '
        '\\textcolor{#RRGGBB}{expression} for colored subexpressions. Runs preserve the '
        'visible transcription and script positions. Read subscripts and superscripts relative to '
        'the baseline in its reading orientation. Use a script only when the smaller glyph is '
        'visibly displaced relative to that baseline; an identifier ending in a full-size digit '
        'remains ordinary text. The baseline is the foot of non-descending letter strokes; '
        'descenders extend below it. Compare digit size with the main letter body, not its descender. '
        'Read bars and accents inside the same locked unit, and keep runs consistent '
        'with LaTeX.',
        'Use resolved when the target attributes are readable. Use ambiguous with units=[] '
        'when the occurrence or its attributes cannot be established from CLEAN, or '
        'excluded_image_text for text belonging to a separate image asset. Give a short '
        'visible observation in reason.'
    ]
    return result


def install():
    from shape_integration.texts import two_stage_crop
    if getattr(two_stage_crop, '_source_attribute_protocol', None) == PROTOCOL:
        return
    original = two_stage_crop.attribute_prompt
    def attribute_prompt(*args, **kwargs):
        return source_attributes(original(*args, **kwargs))
    two_stage_crop.attribute_prompt = attribute_prompt
    two_stage_crop._source_attribute_protocol = PROTOCOL
