"""Comparison views cover actual candidates without expanding to whole peers."""
import numpy as np
import pytest

from full_recovery.quality_choice import comparison_box,validate_source_inventory


def test_context_rail_does_not_expand_a_local_line_view():
    rows=[{'points':np.array([[60.,90.],[700.,90.]]),'endpoint_nodes':['vertical_rail']}]
    assert comparison_box(rows,(912,750))==(42,72,719,109)


def test_full_candidate_and_attached_head_extents_survive_the_crop():
    rows=[{'points':np.array([[60.,90.],[700.,90.]]),
           'attachment_bounds':[[[699.,85.],[710.,90.],[699.,95.]]]},
          {'points':np.array([[699.,90.],[699.,680.]])}]
    box=comparison_box(rows,(912,750))
    assert box==(42,67,729,699)
    assert all(type(v)is int for v in box)


def test_complete_output_envelope_preserves_source_observations_and_strict_guards():
    objects=[{'id':'S1','source_path':[[5,10],[80,10]],'description':'The visible horizontal source stroke.'}]
    raw={'output':{'source_objects':objects}}
    assert validate_source_inventory(raw,[0,0,100,100])=={'source_objects':objects}
    assert raw=={'output':{'source_objects':objects}}
    assert validate_source_inventory({**raw,'validation_error':'','correction':''},[0,0,100,100])=={'source_objects':objects}
    with pytest.raises(ValueError):validate_source_inventory({**raw,'validation_error':'Unresolved failure'},[0,0,100,100])
    with pytest.raises(ValueError):validate_source_inventory({'output':{'source_objects':[]}},[0,0,100,100])
    with pytest.raises(ValueError):validate_source_inventory({'output':{'source_objects':[
        {'id':'S1','source_path':[[0,10],[200,10]],'description':'Outside crop.'}]}},[0,0,100,100])
