"""A split explanatory brace is one occurrence; nearby real braces stay."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from full_recovery.handoff import strokes,resample,split_brace_duplicates


def fixture(tmp_path):
    commands=[['M',20,100],['C',20,70,25,70,40,70],['L',180,70],
              ['C',200,70,210,50,220,50],['C',230,50,240,70,260,70],
              ['L',400,70],['C',420,70,420,70,420,100]]
    lines=strokes(commands);q=resample(lines,400)
    scene={'parts':{'brace':{'object_id':'group','status':'ok','commands':commands}},
           'objects':{'group':{'candidate':{'kind':'leftBrace'}}}}
    styles={'brace':{'stroke':'#000000','fill':None,'width_px':3}}
    rows=[]
    for i,points in enumerate((q[:201],q[200:])):
        shifted=points+np.array([0,4])
        rows.append({'id':str(i),'kind':'curve','physical_target_id':'one_source_brace',
            'start':shifted[0].tolist(),'end':shifted[-1].tolist(),'via':shifted[1:-1].tolist(),
            'heads':['none','none']})
    source=tmp_path/'source.png';image=Image.new('RGB',(450,140),'white')
    ImageDraw.Draw(image).line([tuple(p) for p in q],fill='black',width=2);image.save(source)
    return scene,styles,rows,source,q


def test_two_approximate_arms_do_not_duplicate_one_source_brace(tmp_path):
    scene,styles,rows,source,_=fixture(tmp_path)
    drop,audit=split_brace_duplicates(scene,styles,rows,source)
    assert drop==['0','1']
    assert audit[0]['single_source_stroke_fraction']>=.85


def test_nearby_parallel_source_brace_must_not_be_deduplicated(tmp_path):
    scene,styles,rows,source,q=fixture(tmp_path)
    image=Image.open(source).convert('RGB')
    ImageDraw.Draw(image).line([tuple(p+[0,5]) for p in q],fill='black',width=2);image.save(source)
    assert split_brace_duplicates(scene,styles,rows,source)[0]==[]


def test_partial_or_headed_connectors_keep_their_independent_role(tmp_path):
    scene,styles,rows,source,_=fixture(tmp_path)
    assert split_brace_duplicates(scene,styles,rows[:1],source)[0]==[]
    rows[0]['heads']=['none','triangle']
    assert split_brace_duplicates(scene,styles,rows,source)[0]==[]
    rows[0]['heads']=['none','none'];scene['objects']['group']['candidate']['kind']='path'
    assert split_brace_duplicates(scene,styles,rows,source)[0]==[]
