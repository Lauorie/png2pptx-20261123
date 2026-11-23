"""Render discovery prompts with one shared rules block; keep review separate."""
import json
import re
from pathlib import Path

BLOCKS = json.loads(Path(__file__).with_name('draft_prompt_blocks.json').read_text())
DISCOVERY_SYSTEM = BLOCKS['discovery_system']
REVIEW_SYSTEM = BLOCKS['review_system']
DISCOVERY_RULES = BLOCKS['discovery_rules']


def fill(text, values):
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise ValueError('Missing draft placeholder: ' + key)
        return str(values[key])
    return re.sub(r'\{\{([A-Z_]+)\}\}', replace, text)


def global_prompt(size):
    return fill(BLOCKS['global_user'], {
        'GLOBAL_WIDTH': size[0], 'GLOBAL_HEIGHT': size[1],
        'DISCOVERY_RULES': DISCOVERY_RULES,
    }) + '\n\n' + BLOCKS['discovery_output']


def local_prompt(size, bounds, local_size):
    return fill(BLOCKS['local_user'], {
        'DISCOVERY_RULES': DISCOVERY_RULES,
        'GLOBAL_WIDTH': size[0], 'GLOBAL_HEIGHT': size[1],
        'LOCAL_WIDTH': local_size[0], 'LOCAL_HEIGHT': local_size[1],
        'LOCAL_REGION_IN_GLOBAL_PX': json.dumps(bounds, ensure_ascii=False),
    }) + '\n\n' + BLOCKS['discovery_output']


def review_prompt(size, rows, boundary_pass=False):
    task = BLOCKS['review_user']
    if boundary_pass:
        task = BLOCKS['boundary_task'] + task[task.index('\n'):]
    return fill(task, {'LOCAL_WIDTH': size[0], 'LOCAL_HEIGHT': size[1],
                       'CANDIDATES_JSON': json.dumps(rows, ensure_ascii=False, indent=2)}) + '\n\n' + BLOCKS['review_output']
