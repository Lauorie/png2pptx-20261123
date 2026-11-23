from copy import deepcopy
from full_recovery.math_alphabets import preview_row


def test_fallback_bold_letter_is_portable_without_changing_recognition_or_native_formula():
    row={'text':'𝐊=','latex':r'\mathbf{K}=','font':'Arial','bold':True,'italic':False,
         'bbox_2d':[20,30,60,50],'runs':[{'text':'𝐊=','script':'normal','color':'#336699'}]}
    before=deepcopy(row);preview,changes=preview_row(row)
    assert row==before and preview['latex']==row['latex'] and preview['bbox_2d']==row['bbox_2d']
    assert preview['text']=='K=' and preview['runs'][0]['bold'] and not preview['runs'][0]['italic']
    assert preview['runs'][1]=={'text':'=','script':'normal','color':'#336699'}
    assert changes==[{'before':'𝐊','after':'K','style':'b'}]


def test_subscripts_and_other_math_alphabets_preserve_their_source_semantics():
    row={'text':'𝑥𝜶𝑲𝒦𝕂+','latex':'unchanged','runs':[
        {'text':'𝑥𝜶𝑲','script':'sub','color':'#000000'},
        {'text':'𝒦𝕂+','script':'normal','italic':False}]}
    preview,changes=preview_row(row)
    assert preview['text']=='xαK𝒦𝕂+' and preview['runs'][0]['script']=='sub'
    assert preview['runs'][0]['italic'] and not preview['runs'][0]['bold']
    assert preview['runs'][1]['italic'] and preview['runs'][1]['bold']
    assert preview['runs'][-1]==row['runs'][-1]
