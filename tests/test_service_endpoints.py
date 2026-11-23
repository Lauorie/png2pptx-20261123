from full_recovery.service_endpoints import sam_base,image_edit_base
from full_recovery.image_edit import Editor


def test_full_action_urls_and_bases_address_the_same_services():
    assert sam_base('http://test:18103/v1/segment/')=='http://test:18103'
    assert sam_base('http://test:18103/v1')=='http://test:18103'
    assert sam_base('http://test:18103')=='http://test:18103'
    assert image_edit_base('http://test:8091/v1/images/edits/')=='http://test:8091/v1'
    assert image_edit_base('http://test:8091/v1')=='http://test:8091/v1'
    assert Editor({'base_url':'http://test:8091/v1/images/edits'}).config['base_url']=='http://test:8091/v1'
