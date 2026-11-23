import numpy as np
from shape_integration.texts.foreground import contrast_ink


def test_colored_node_background_is_not_text_ink():
    rgb=np.full((50,100,3),255,dtype=np.float32)
    rgb[8:42,10:90]=[190,220,180]
    rgb[20:30,40:50]=[0,0,0]
    ink=contrast_ink(rgb,[30,14,60,36])
    assert ink[25,45]>.9 and ink[15,45]==0
    assert ink[2,45]<.4


def test_white_and_gray_text_do_not_require_color_prior():
    for background,foreground in [(20,255),(255,180)]:
        rgb=np.full((40,60,3),background,dtype=np.float32)
        rgb[15:25,25:35]=foreground
        ink=contrast_ink(rgb,[15,5,45,35])
        assert ink[20,30]>.9 and ink[6,16]==0


def test_sparse_text_below_five_percent_is_not_blank():
    rgb=np.full((100,100,3),255,dtype=np.float32)
    rgb[40:47,20:90]=0  # 4.9% support, previously discarded by P95.
    ink=contrast_ink(rgb,[0,0,100,100])
    assert np.count_nonzero(ink)==490
    assert ink[43,50]>.9 and ink[0,0]==0


def test_blank_and_low_contrast_regions_remain_blank():
    for background in (0,180,255):
        rgb=np.full((100,100,3),background,dtype=np.float32)
        assert not contrast_ink(rgb,[0,0,100,100]).any()
    rgb=np.full((100,100,3),255,dtype=np.float32)
    rgb[40:47,20:90]=250
    assert not contrast_ink(rgb,[0,0,100,100]).any()


def test_one_percent_ink_is_not_discarded_by_area_fraction():
    rgb=np.full((100,100,3),255,dtype=np.float32)
    rgb[45:55,45:55]=0
    ink=contrast_ink(rgb,[0,0,100,100])
    assert np.count_nonzero(ink)==100


def test_extra_blank_padding_does_not_erase_sparse_ink():
    small=np.full((20,20,3),255,dtype=np.float32)
    small[8:12,5:15]=0
    large=np.full((200,200,3),255,dtype=np.float32)
    large[90:110,90:110]=small
    a=contrast_ink(small,[0,0,20,20])
    b=contrast_ink(large,[0,0,200,200])
    assert np.array_equal(a,b[90:110,90:110])
    assert np.count_nonzero(b)==40  # Only 0.1% of the evaluation region.
