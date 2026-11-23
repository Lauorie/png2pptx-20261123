from copy import deepcopy
from full_recovery.text_angle_transport import normalize


def test_equivalent_angles_preserve_content_and_do_not_mutate_reply():
    raw={'items':[{'id':'label','status':'resolved','units':[{'rotation_deg':270,'text':'Visible label','layout_mode':'single_line'}]},
                  {'id':'flat','status':'located','rotation_deg':450,'text':'Other label'}]}
    old=deepcopy(raw);out=normalize(raw)
    assert out['items'][0]['units'][0]['rotation_deg']==-90
    assert out['items'][1]['rotation_deg']==90
    assert out['items'][0]['units'][0]['text']=='Visible label' and raw==old
    assert normalize(out)==out


def test_invalid_angles_remain_for_strict_validator_to_reject():
    raw={'items':[{'rotation_deg':'270'},{'rotation_deg':True},{'rotation_deg':None},{'rotation_deg':float('inf')}]}
    out=normalize(raw)
    assert out==raw
