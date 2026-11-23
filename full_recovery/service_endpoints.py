"""Environment-selected services for newly prepared workflows."""
import os
SAM_BASE_URL = os.environ.get('PPT_SAM_URL', 'http://127.0.0.1:18103')
IMAGE_EDIT_BASE_URL = os.environ.get('PPT_IMAGE_EDIT_URL', 'http://127.0.0.1:8091/v1')

def sam_base(url):
    value = url.rstrip('/')
    for suffix in ('/v1/segment', '/v1'):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value

def image_edit_base(url):
    value = url.rstrip('/')
    return value[:-len('/images/edits')] if value.endswith('/images/edits') else value
