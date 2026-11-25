"""Complete PPT recovery using current shape, point-geometry arrows and GLM text."""
import argparse
from pathlib import Path
import sys
from full_recovery.service_endpoints import SAM_BASE_URL

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT/'backends/reference'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--input-root',type=Path);p.add_argument('--ids',nargs='+')
    p.add_argument('--text-source',type=Path,default=EXPERIMENTS/'shape_text_glm_geometry_20261019')
    p.add_argument('--arrow-reference',type=Path,default=EXPERIMENTS/'arrow_mask_first_20261016')
    p.add_argument('--arrow-code-run',type=Path,help='Freeze only this geometry-first implementation; run all branches fresh')
    p.add_argument('--arrow-products-run',type=Path,help='Use explicitly selected existing arrow results; infer shape and text fresh')
    p.add_argument('--pending-arrow-ids',nargs='+',default=[],help='Generate shape/text for these explicitly deferred missing arrow cases; leave composition pending')
    p.add_argument('--arrow-implementation',type=Path,help='Pin this geometry_code implementation independently of reused arrow products')
    p.add_argument('--arrow-delivery',type=Path,default=EXPERIMENTS/'arrow_method/delivery',
                   help='Method/audit reference only; previous recovered artifacts are never imported')
    p.add_argument('--api',action='store_true',help='Prepare CPU-only hosted-GLM + shared-SAM batch')
    p.add_argument('--shape-only',action='store_true',help='Run shape recovery only; preserve checkpoints for later composition')
    p.add_argument('--api-key-file',type=Path,action='append',help='Protected credential file; repeat for multiple keys')
    p.add_argument('--api-base-url',default='http://35.220.164.252:3888/v1')
    p.add_argument('--api-model',default='glm-5.3-flash')
    p.add_argument('--force-reasoning-effort',choices=('low','medium','high'))
    p.add_argument('--arrow-reasoning-effort',choices=('low','medium','high'),
                   help='Reasoning effort for the arrow branch only; overrides --force-reasoning-effort there')
    p.add_argument('--arrow-max-tokens',type=int,help='Output cap per arrow-branch request')
    p.add_argument('--critic-profile',choices=('full','lean'),default='full')
    p.add_argument('--api-budget-config',type=Path,help='Shared persistent task budget configuration')
    p.add_argument('--sam-url',default=SAM_BASE_URL)
    p.add_argument('--namespace',default='ailab-intern10')
    p.add_argument('--charged-group',default='intern10_proxy')
    p.add_argument('--worker-cpu',type=int,default=16)
    p.add_argument('--worker-memory-mib',type=int,default=65536)
    p.add_argument('--api-concurrency',type=int,default=48)
    p.add_argument('--case-workers',type=int,help='Concurrent source cases in the CPU workflow')
    p.add_argument('--text-workers',type=int,help='Concurrent original-image text cases')
    p.add_argument('--image-edit-url',help='Optional SenseNova /v1 endpoint for the shared compositor')
    p.add_argument('--image-edit-model',default='SenseNova-U1.5-8B-MoT')
    p.add_argument('--image-edit-concurrency',type=int,default=2)
    p.add_argument('--arrow-native-renderer',type=Path,help='Validated 26.8 executable for geometry-first arrow previews')
    p.add_argument('--native-renderer',type=Path,help='Explicit LibreOffice executable for shape, text and final PPT renders')
    args=p.parse_args()
    if args.arrow_code_run and args.arrow_products_run:
        p.error('Choose either fresh arrow inference or explicitly selected arrow products')
    if args.pending_arrow_ids and not args.arrow_products_run:
        p.error('--pending-arrow-ids requires --arrow-products-run')
    if args.shape_only and args.prepare and not args.api:p.error('--shape-only requires --api')
    if args.shape_only and any((args.arrow_code_run,args.arrow_products_run,args.arrow_implementation)):
        p.error('--shape-only does not accept arrow inputs; compose them later')
    if args.prepare:
        if args.input_root is None:p.error('--prepare requires --input-root')
        from full_recovery.prepare import prepare
        prepare(ROOT,args)
        if args.api:
            from full_recovery.prepare_api import configure
            configure(ROOT,args)
    else:
        import json
        settings=json.loads((args.run_dir/'settings.json').read_text())
        if settings.get('workflow_scope')=='fresh_full':
            from full_recovery.fresh_pipeline import execute
        else:
            from full_recovery.pipeline import execute
        execute(args.run_dir.resolve())


if __name__=='__main__':main()
