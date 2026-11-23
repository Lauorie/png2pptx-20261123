"""Native circle geometry must reach both SAM ellipse-family labels."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment/style_line_sam'))
from sam_tiles import prompts_for,masks_from,STYLE_CATEGORIES
from shape_integration.styling.sam_lines import select_targets


def test_native_ellipse_queries_circle_and_ellipse_without_a_ratio_cutoff():
    assert prompts_for('ellipse','both')==[{'name':'line'},{'name':'circle'},{'name':'ellipse'}]
    assert prompts_for('ellipse','class')==[{'name':'circle'},{'name':'ellipse'}]
    assert prompts_for('ellipse','line')==[{'name':'line'}]
    assert prompts_for('path','both')==[{'name':'line'},{'name':'polygon'}]


def test_circle_mask_reaches_normal_contour_matching_but_neighbour_is_rejected(tmp_path):
    y,x=np.indices((40,40));circle=(x-20)**2+(y-20)**2<=6**2
    Image.fromarray(circle.astype('uint8')*255).save(tmp_path/'circle.png')
    neighbour=(x-4)**2+(y-4)**2<=3**2
    Image.fromarray(neighbour.astype('uint8')*255).save(tmp_path/'neighbour.png')
    result={'predictions':[{'id':name,'category':'circle','score':.9,'mask_path':str(tmp_path/(name+'.png'))}
                           for name in ['circle','neighbour']]}
    distance=abs(np.hypot(x-20,y-20)-6)
    filtered=masks_from(result,.5,['line','rectangle','ellipse','polygon','other'])
    assert select_targets(filtered,distance,1.5,1)[0]is None
    actual=masks_from(result,.5,STYLE_CATEGORIES)
    union,audit=select_targets(actual,distance,1.5,1)
    assert union is not None and np.array_equal(union,circle)
    assert [r['id']for r in audit if r['target']]==['circle']
