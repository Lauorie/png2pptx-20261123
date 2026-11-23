"""Prompt v004: v003 plus one generic endpoint-direction consistency rule.

Promoted unchanged from the head_direction_consistency_20261105_01 study.
The prompt and system text match that experiment; this module sends no requests.
"""

SYSTEM = (
    'Recover visible line-end styles. Return only the requested JSON object.'
)

VERSION = 'head_prompt_v004_direction_consistency_20261105'


DIRECTION_RULE = (
    'Round dots may be shared; recover an arrowhead according to the local direction '
    'at the current path endpoint so that its rendered pointing direction and the side '
    'occupied by its wings match SOURCE ORIGINAL; if the pointing direction does not '
    'match, choose none.'
)


def prompt(row, tag, size):
    return {
        'task': (
            'Reproduce SOURCE ORIGINAL at each fixed endpoint; '
            'do not infer the original file\'s object structure.'
        ),
        'size': list(size),
        'target': {
            'id': row['id'],
            'start': row['start'],
            'end': row['end'],
            'locator_tags': {'start': tag + 'S', 'end': tag + 'E'},
        },
        'categories': {
            'none': (
                'No visible line head: plain cap, bare junction, or contact '
                'with a separate node.'
            ),
            'circle': (
                'Filled round dot joining the shaft at this endpoint, matching '
                'its color/fill. Shared junction dots count too.'
            ),
            'arrow': 'Visible pointed head attached to this shaft here.',
        },
        'instructions': [
            DIRECTION_RULE,
            (
                f'{tag}S = start; {tag}E = end. Use bracket centers, not labels. '
                'LOCATORS colors, brackets and dashed leaders are annotations.'
            ),
            (
                'Nodes with their own fill, outline or content stay separate. '
                'Touching their edge is not a circle head. Check attachment '
                'as well as color.'
            ),
            (
                'Inspect SOURCE at the marked point. Do not infer heads from '
                'expected flow, nearby arrows, borders or blur.'
            ),
            (
                'Coordinates bind start/end, not flow. Geometry, shaft style '
                'and head size stay fixed.'
            ),
            (
                'Optional REFERENCE shows generic right-end categories, '
                'not the target or a size standard.'
            ),
        ],
        'output_schema': {
            'id': row['id'],
            'start': 'none|circle|arrow',
            'end': 'none|circle|arrow',
        },
    }
