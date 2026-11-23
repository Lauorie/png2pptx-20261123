from copy import deepcopy
import numpy as np
import pytest
from lxml import etree

from full_recovery.quality_choice import candidates,nominate,apply_choices
from full_recovery.compose import selection_subset,read_package,SLIDE,P,A
from shape_integration.fitting.scene import write_pptx


@pytest.mark.parametrize('angle,scale',[(0,1),(90,1),(32,2.5),(180,.6)])
def test_exact_heads_keep_native_subpath_indices_without_editing_siblings(tmp_path,angle,scale):
    theta=np.radians(angle);m=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*scale
    xy=np.array([[10,20],[70,20],[62,16],[70,20],[62,24],
                 [10,50],[70,50],[62,46],[70,50],[62,54]],float)@m.T+[180,160]
    commands=[[op,*p] for op,p in zip(['M','L','M','L','L']*2,xy)]
    part={'id':'routes','object_id':'owner','kind':'path','status':'ok','commands':commands,
          'bbox_2d':[*xy.min(0),*xy.max(0)],'native_parameters_original':None}
    scene={'source_size':[500,500],'parts':{'routes':part},
           'objects':{'owner':{'id':'owner','state':'accepted_by_model','part_keys':['routes']}}}
    before=deepcopy(scene)
    rows=candidates(scene,{'routes':{'stroke':'#000000','width_px':scale,'fill':None}},[])
    assert len(rows)==2 and {tuple(r['subpaths']) for r in rows}=={(0,1),(2,3)}
    ppt=tmp_path/'source.pptx';selected=tmp_path/'evidence.pptx';write_pptx(scene,ppt)
    selection_subset(ppt,selected,rows[:1])
    original=etree.fromstring(read_package(ppt)[SLIDE]);shown=etree.fromstring(read_package(selected)[SLIDE])
    paths=shown.findall('.//{'+A+'}custGeom/{'+A+'}pathLst/{'+A+'}path')
    assert len(paths)==1 and len(paths[0].findall('{'+A+'}moveTo'))==2
    assert etree.tostring(shown.find('.//{'+A+'}ln'))==etree.tostring(original.find('.//{'+A+'}ln'))
    assert scene==before
    plan=apply_choices(rows,[{'candidate_ids':[rows[0]['id']],'keep_ids':[]}])
    assert plan['shape_subpath_suppression']=={'owner/routes':[0,1]}

