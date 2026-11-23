from copy import deepcopy
import numpy as np
from PIL import Image
from shape_integration.texts import registration, colors


def test_original_anchor_selected_only_with_clear_pixel_advantage(monkeypatch):
    row={'id':'X','bbox_2d':[50,50,70,80],'geometry_prior_bbox':[10,10,30,40]}
    def check(r,m,im):
        prior=r['bbox_2d']==row['geometry_prior_bbox']
        return r,m,{'status':'accepted' if prior else 'low_score','after':{'score':.9 if prior else .5}}
    monkeypatch.setattr(registration,'_register_at',check)
    result,_,audit=registration.register(row,{},None)
    assert result['bbox_2d']==row['geometry_prior_bbox']
    assert audit['anchor']=='original_occurrence' and row['bbox_2d']==[50,50,70,80]


def test_uniform_formula_color_uses_verified_ink_and_preserves_native_structure(tmp_path):
    rgba=np.zeros((24,20,4),dtype=np.uint8);rgba[4:20,5:15,3]=255
    preview=tmp_path/'formula.png';Image.fromarray(rgba).save(preview)
    image=np.full((50,50,3),255,dtype=np.uint8);image[12:20,12:18]=0
    row={'id':'X','kind':'formula','text':'x1','latex':'x_1','color':'#B0B0B0',
         'bbox_2d':[10,10,20,22],'frame_px':[10,10,10,12],'font_size_px':24,'rotation_deg':0,
         'runs':[{'text':'x','script':'normal'},{'text':'1','script':'sub'}],
         'glyph_registration':{'status':'accepted','score':.9},
         'layout':{'measurement':{'reference_size_px':24,'width':10,'height':12,'equation_preview':str(preview)}}}
    before=preview.read_bytes()
    fixed,audit=colors.recover_formula(deepcopy(row),image,tmp_path/'assets')
    assert fixed['color']=='#000000' and fixed['latex']=='x_1'
    assert [r['script'] for r in fixed['runs']]==['normal','sub']
    assert preview.read_bytes()==before and audit['policy']=='pixel_calibrated_uniform_formula'
    row['glyph_registration']['score']=.4
    kept,_=colors.recover_formula(deepcopy(row),image,tmp_path/'unused')
    assert kept['color']=='#B0B0B0'
