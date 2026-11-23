import json
from pathlib import Path
import pytest
from full_recovery.prepare import sha
from full_recovery.text_product_validation import verify


def product(tmp_path):
    original=tmp_path/'original.png';original.write_bytes(b'original raster')
    item={'id':'same_source','source_sha256':sha(original),'original':str(original),'source_size':[100,80]}
    output=tmp_path/'case';final=output/'final';final.mkdir(parents=True)
    # The scene intentionally has the actual reference schema: no source hash.
    (final/'scene.json').write_text(json.dumps({'texts':[],'source_size':[100,80]}))
    (final/'provenance.json').write_text(json.dumps({'image_sha256':sha(original),'review_policy':'effective'}))
    (output/'provenance.json').write_text(json.dumps({'parameters':{'image_sha256':sha(original),
        'baseline_run':None,'base_pptx_sha256':None}}))
    for name in ['text.pptx','native.pdf','native.png','quality.json']:(final/name).write_bytes(b'completed native bytes')
    return item,output


def test_source_hash_is_read_from_provenance_not_scene(tmp_path):
    item,output=product(tmp_path);result=verify(item,output)
    assert result['source_sha256']==item['source_sha256']
    assert result['effective_review_policy']=='effective'
    assert len(result['products_sha256'])==6


def test_other_source_provenance_cannot_authorize_same_size_text(tmp_path):
    item,output=product(tmp_path)
    (output/'final/provenance.json').write_text(json.dumps({'image_sha256':'different_source'}))
    with pytest.raises(ValueError,match='provenance differs'):verify(item,output)


def test_original_only_text_rejects_a_historical_baseline(tmp_path):
    item,output=product(tmp_path)
    (output/'provenance.json').write_text(json.dumps({'parameters':{'image_sha256':item['source_sha256'],
        'baseline_run':'old_prediction_product'}}))
    with pytest.raises(ValueError,match='prior baseline'):verify(item,output)
