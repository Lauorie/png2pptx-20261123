"""A short visual call writes the actual edit prompt from the expanded source crop."""
import re

from PIL import Image

from ..frames import Crop
from ..scene import save
from .clients import cached_json_call
from .prompts import IMAGE_LABELS, PROMPT_WRITER_SYSTEM, sensenova_prompt, visual_edit_prompt


class NeedsContext(ValueError):
    """The visual writer needs one wider source crop before any image edit."""


def validate_plan(raw, size):
    if raw.get('decision') not in ('ready', 'needs_context', 'unresolved'):
        raise ValueError('Invalid visual prompt decision')
    if not isinstance(raw.get('uncertain'), list): raise ValueError('Missing prompt uncertainty')
    if raw['decision'] != 'ready': return raw
    if raw.get('background') not in ('alpha', 'opaque_panel'): raise ValueError('Invalid planned background')
    for key in ('target', 'keep', 'remove', 'edit_prompt'):
        if not isinstance(raw.get(key), str) or not raw[key].strip() or len(raw[key]) > 1800:
            raise ValueError('Missing/oversized visual edit field: ' + key)
    if re.search(r'\b[xy]\s*(?:[=：:]|[0-9])|\bbbox\b|\[\s*\d+(?:\.\d+)?\s*,\s*\d+', raw['edit_prompt'], re.I):
        raise ValueError('edit_prompt must use visual descriptions without pixel coordinates: the writer and editor have different image sizes. Detection owns the bbox; do not output coordinates here.')
    return raw


def plan_edit(client, source, asset, frame, folder):
    if asset.get('edit_feedback') or asset.get('previous_edit_path'):
        raise ValueError('Manual edit feedback is not allowed in the automatic image pipeline')
    context = tuple(frame['context_box_source_half_open'])
    crop = Crop(tuple(source.size), context, tuple(frame['writer_input_size']))
    box = asset['source_box_half_open']
    hint = crop.to_local(box[:2])+crop.to_local([box[2]-1, box[3]-1])
    prompt = visual_edit_prompt(asset, hint, crop.input_size, frame['key_rgb'])
    images = [(IMAGE_LABELS['prompt_writer'], folder/'writer_input_2048.png')]
    result, evidence = cached_json_call(client, 'prompt_'+asset['id'], PROMPT_WRITER_SYSTEM, prompt, images,
                                       lambda r: validate_plan(r, crop.input_size))
    save(folder/'prompt_plan.json', {**result, 'call': evidence})
    if result['decision'] == 'needs_context':
        raise NeedsContext('Visual prompt needs inspection: ' + str(result))
    if result['decision'] != 'ready':
        raise ValueError('Visual prompt needs inspection: ' + str(result))
    updated = {**asset, **{k: result[k] for k in ('target', 'keep', 'remove', 'background')},
               'provenance': {**asset.get('provenance', {}), 'prompt_writer_call': evidence,
                              'bbox_authority': 'detector', 'writer_uncertain': result['uncertain']}}
    px, py = frame['source_to_edit']['padding_xy']
    source.crop(box).save(folder/'reviewed_source_crop.png')
    # The source canvas is unchanged; only padded pixels follow the reviewed background mode.
    canvas = Image.new('RGB', tuple(frame['edit_input_size']), tuple(frame['key_rgb']) if result['background']=='alpha' else 'white')
    canvas.paste(source.crop(context).resize(tuple(frame['resized_context_size']), Image.Resampling.LANCZOS), (px, py))
    canvas.save(folder/'input.png'); save(folder/'transform.json', frame)
    return updated, sensenova_prompt(result['edit_prompt'])
