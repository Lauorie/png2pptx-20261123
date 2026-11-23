import json

import pytest

from run_shapes import input_records


def write_catalog(path, package_root, image_rel, ident='case-a'):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = package_root/image_rel
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b'png fixture')
    path.write_text(json.dumps({'items':[{'case_key':ident,'image':image_rel}]}))


def test_catalog_at_package_root_keeps_stable_case_key(tmp_path):
    write_catalog(tmp_path/'catalog.json',tmp_path,'images/case.png')
    records=input_records(tmp_path)
    assert len(records)==1
    assert records[0][0]=='case-a'
    assert records[0][1]==(tmp_path/'images/case.png').resolve()


def test_nested_catalog_resolves_paths_from_final_package_root(tmp_path):
    write_catalog(tmp_path/'conference150/catalog.json',tmp_path,'conference150/cases/case-a/original.png')
    records=input_records(tmp_path)
    assert records[0][1]==(tmp_path/'conference150/cases/case-a/original.png').resolve()


def test_input_root_can_point_at_catalog_directory(tmp_path):
    write_catalog(tmp_path/'conference150/catalog.json',tmp_path,'conference150/cases/case-a/original.png')
    records=input_records(tmp_path/'conference150')
    assert records[0][1]==(tmp_path/'conference150/cases/case-a/original.png').resolve()


def test_catalog_id_filter_rejects_unknown_case(tmp_path):
    write_catalog(tmp_path/'catalog.json',tmp_path,'images/case.png')
    assert input_records(tmp_path,['case-a'])[0][0]=='case-a'
    with pytest.raises(ValueError,match='Unknown catalog cases'):
        input_records(tmp_path,['missing'])
