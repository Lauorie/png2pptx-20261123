"""Environment-selected services for newly prepared workflows."""
import os
from typing import Any, Dict, List, Sequence
SAM_BASE_URL = os.environ.get('PPT_SAM_URL', 'http://127.0.0.1:18103')
IMAGE_EDIT_BASE_URL = os.environ.get('PPT_IMAGE_EDIT_URL', 'http://127.0.0.1:8091/v1')

def sam_base(url):
    value = url.rstrip('/')
    for suffix in ('/v1/segment', '/v1'):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value

def sam_bases(value: str) -> List[str]:
    """Return the SAM instances named by a comma-separated list of addresses.

    One instance segments one image at a time, so the shape branch's line
    sampling queues behind it. Several instances of the same model on one GPU
    take the requests side by side.

    Args:
        value: One address, or several separated by commas.

    Returns:
        The normalised addresses in the order given, each once.

    Raises:
        ValueError: When no address is given.
    """
    result: List[str] = []
    for part in value.split(','):
        address = sam_base(part.strip())
        if address and address not in result:
            result.append(address)
    if not result:
        raise ValueError('No SAM service address given')
    return result


def same_sam_model(profiles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the first instance's identity after checking that every instance serves the same weights.

    Args:
        profiles: One verified identity per instance: ``url``, ``epoch``, ``sha256``.

    Returns:
        The first profile.

    Raises:
        ValueError: When an instance serves another checkpoint, naming it.
    """
    first = profiles[0]
    for other in profiles[1:]:
        if (other['epoch'], other['sha256']) != (first['epoch'], first['sha256']):
            raise ValueError('SAM instance %s serves a different checkpoint than %s' % (other['url'], first['url']))
    return first


def image_edit_base(url):
    value = url.rstrip('/')
    return value[:-len('/images/edits')] if value.endswith('/images/edits') else value
