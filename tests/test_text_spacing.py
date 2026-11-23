from copy import deepcopy
from full_recovery.text_spacing import normalize


def test_single_baseline_uses_the_same_spacing_in_runs_and_measurement():
    row={'text':'(i)\tSignal','kind':'text','layout_mode':'single_line','visual_line_count':1,
         'bbox_2d':[10,20,70,35],'runs':[{'text':'(i)\tSignal','bold':False}]}
    before=deepcopy(row);updated=normalize(row)
    assert updated['text']=='(i) Signal' and updated['runs'][0]['text']==updated['text']
    assert updated['bbox_2d']==row['bbox_2d'] and row==before


def test_layout_structure_and_existing_spaces_do_not_change():
    for row in [{'text':'A\tB','kind':'formula'},{'text':'A\tB','layout_mode':'multiline'},
                {'text':'A\tB','visual_line_count':2},{'text':'A  B','kind':'text'}]:
        assert normalize(row) is row
