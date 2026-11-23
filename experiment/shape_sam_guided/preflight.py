"""Bounded local verification; no model loading or inference."""
import argparse
import ast
import io
from pathlib import Path
import unittest
from core import HERE, CODE_ROOT, read, save, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=read(args.config)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),str(p))
    for name in ['shape.py','sam_evidence.py','fitting/local.py','fitting/control.py','fitting/critic.py']:
        p=CODE_ROOT/'src/shape_integration'/name;ast.parse(p.read_text(),str(p))
    import test_forward, test_guidance, test_fixed
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in [test_forward,test_guidance,test_fixed]])
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    print(log.getvalue(),flush=True)
    base=Path(config['baseline_root']);rows=[]
    for item in read(base/'inputs/manifest.json')['images']:
        case=base/'runs'/config['baseline_run']/'cases'/item['id']
        paths=[base/item['original'],base/item['input'],case/'fitting/scene.json',case/'shapes/global_raw.json',case/'shapes/plan.json',case/'dedup/image_boxes_original.json']
        hashes={str(p):sha(p) for p in paths}
        if hashes[str(base/item['original'])]!=item['source_sha256']:raise ValueError('Source changed')
        rows.append({'id':item['id'],'sha256':hashes})
    save(args.output,{'passed':result.wasSuccessful(),'tests':result.testsRun,'baseline_inputs':rows,
         'source_files_reused_in_place':True,'weight_training':False})
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
