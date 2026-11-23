import numpy as np
from shape_integration.styling.sam_lines import plan_tiles


def test_dense_shared_lattice_covers_every_route_with_interior_margin():
    commands=[];points=[]
    for y in range(20,260,13):
        for z in range(30,250,17):
            commands += [['M',10,y],['L',390,z]]
            points.extend(np.array([10,y])+(np.array([390,z])-np.array([10,y]))*
                          np.linspace(0,1,100)[:,None])
    boxes=plan_tiles({'commands':commands},[400,280],96)
    assert len(boxes)<100
    samples=np.array(points)
    covered=np.zeros(len(samples),bool)
    for x,y,s in boxes:
        covered|=((samples>=[x+16,y+16])&(samples<=[x+s-16,y+s-16])).all(1)
    assert covered.all()


def test_sparse_and_independent_short_contours_keep_existing_tile_locations():
    part={'commands':[['M',10,20],['L',60,20]]}
    assert plan_tiles(part,[100,100],96)==[[-12,-28,96]]
    part={'commands':[['M',10,20],['L',60,20],['M',200,20],['L',250,20]]}
    assert plan_tiles(part,[300,100],96)==[[-12,-28,96],[178,-28,96]]


def test_tile_clipping_retains_exact_raster_sampling_phase():
    from scipy.ndimage import distance_transform_edt
    from shape_integration.styling.sam_lines import path_band,to_tile
    from shape_integration.styling.lines import subpaths,polyline
    rng=np.random.default_rng(7)
    commands=[]
    for a,b in rng.uniform(-500,1500,(60,2,2)):
        commands += [['M',*a],['L',*b]]
    part={'commands':commands};box=[150,250,96];size=128
    line=np.zeros((size,size),bool)
    for sub in subpaths(commands):
        q=to_tile(polyline(sub),box,size)
        for a,b in zip(q[:-1],q[1:]):
            n=max(2,int(np.ceil(np.linalg.norm(b-a)))+1)
            p=np.rint(a+(b-a)*np.linspace(0,1,n)[:,None]).astype(int)
            ok=((p>=0)&(p<size)).all(1);line[p[ok,1],p[ok,0]]=True
    expected=distance_transform_edt(~line)*box[2]/size
    assert np.array_equal(path_band(part,box,size,2),expected)


def test_changed_aggregate_protocol_preserves_exact_source_tiles(tmp_path,monkeypatch):
    import importlib.util,json,hashlib
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('sam_checkpoint_test',root/'experiment/style_line_sam/sam_tiles.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source.png';Image.new('RGB',(80,60),'white').save(source)
    scene=tmp_path/'scene.json';scene.write_text(json.dumps({'parts':{},'source_size':[80,60]}))
    config={'side':96,'prompts':'both','categories':[],'score_threshold':.5}
    folder=tmp_path/'out'/'generic';folder.mkdir(parents=True)
    identity={'source':hashlib.sha256(source.read_bytes()).hexdigest(),'scene':hashlib.sha256(scene.read_bytes()).hexdigest(),
              'config':config,'sam':dict.fromkeys(('sam_threshold','external_sam_epoch','external_sam_checkpoint_sha256','sam_checkpoint_sha256')),
              'protocol':'stable_line_evidence.1'}
    old={'identity':identity,'parts':{},'sam_requests':0,'sam_wall_seconds':0}
    (folder/'line_evidence.json').write_text(json.dumps(old))
    tiles=folder/'tiles/unchanged';tiles.mkdir(parents=True);(tiles/'tile.png').write_bytes(source.read_bytes())
    monkeypatch.setattr(module,'scene_parts',lambda scene:{})
    def sam(*args):raise AssertionError('No contours require SAM requests')
    with ThreadPoolExecutor(max_workers=1)as pool:
        result=module.run_case({'id':'generic','original':str(source),'scene':str(scene)},config,tmp_path/'out',sam,pool)
    assert result['tiles']==0
    assert (tiles/'tile.png').read_bytes()==source.read_bytes()
    archived=list((folder/'previous_evidence').glob('*/line_evidence.json'))
    assert len(archived)==1 and json.loads(archived[0].read_text())==old
    assert json.loads((folder/'line_evidence.json').read_text())['identity']['protocol']=='shared_source_tile_line_evidence.2'
