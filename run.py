"""Run localization only or the integrated experimental fitting pipeline."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read, Runtime
from shape_integration.batch import run
from png2pptx.api.relay_client import load_bashrc_credentials

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);p.add_argument('--ids',nargs='+')
    p.add_argument('--with-fitting',action='store_true');p.add_argument('--from-localization',type=Path)
    p.add_argument('--with-styling',action='store_true',help='Recover styles after final geometry, including one style critic/refine cycle')
    p.add_argument('--with-arrows',action='store_true',help='Append thin native arrow recovery after the shape pipeline')
    p.add_argument('--with-texts',action='store_true',help='Append line OCR and editable text after shapes and optional arrows')
    p.add_argument('--with-images',action='store_true',help='Embed detected image crops below native text/formulas')
    p.add_argument('--complete',action='store_true',help='Run fitting, styles, arrows, image insertion and text in one pipeline')
    p.add_argument('--dedup-only',action='store_true',help='Reuse saved localization and rerun only the shape/image conflict filter')
    p.add_argument('--from-fitting',type=Path,help='Reuse saved geometry for a fresh full critic/refinement run')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--selection',type=Path)
    p.add_argument('--config',type=Path,help='Independent run configuration; defaults to config.json')
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):raise ValueError('Run name must be one path component')
    config=read(args.config if args.config is not None else ROOT/'config.json');items=read(ROOT/'inputs/manifest.json')['images']
    config['styling_enabled']=bool(config.get('styling_enabled',False) or args.with_styling or args.complete)
    config['arrows_enabled']=bool(config.get('arrows_enabled',False) or args.with_arrows or args.complete)
    config['texts_enabled']=bool(config.get('texts_enabled',False) or args.with_texts or args.complete)
    config['image_export_enabled']=bool(config.get('image_export_enabled',False) or args.with_images or args.complete)
    if args.dedup_only and config['image_export_enabled']:raise ValueError('Dedup-only cannot include image composition')
    if args.dedup_only and config['texts_enabled']:raise ValueError('Dedup-only cannot include texts')
    if args.dedup_only and config['arrows_enabled']:raise ValueError('Dedup-only cannot include arrows')
    if args.dedup_only and config['styling_enabled']:raise ValueError('Dedup-only cannot include styling')
    if args.dedup_only and (args.from_localization is None or args.with_fitting or args.from_fitting or args.selection):
        raise ValueError('--dedup-only requires --from-localization and cannot be combined with fitting/selection')
    config['fitting_enabled']=not args.dedup_only and (config['fitting_enabled'] or args.with_fitting or config['styling_enabled'] or config['arrows_enabled'] or config['texts_enabled'] or config['image_export_enabled'] or args.from_localization is not None or args.from_fitting is not None)
    if args.from_fitting is not None and (args.from_localization is not None or args.selection):
        raise ValueError('--from-fitting reuses all saved candidates; do not combine with localization/selection')
    if config['manual_overrides']:raise ValueError('Manual image dedup overrides are not supported')
    if args.selection:
        if not config['fitting_enabled']:raise ValueError('Selection requires fitting')
        selected=read(args.selection)['examples'];config['fit_selection']=[{'case_id':x['case_id'],'object_id':x['object_id']} for x in selected]
        items=[o for o in items if o['id'] in {x['case_id'] for x in selected}]
        if set(x['case_id'] for x in selected)!={o['id'] for o in items}:raise ValueError('Unknown selected case')
        config['fitting']['coverage_rounds']=0
    if args.ids:
        if not set(args.ids).issubset({o['id'] for o in items}):raise ValueError('Unknown case')
        items=[o for o in items if o['id'] in args.ids]
    runtime=None
    if not args.prepare_only:
        base,key,_=load_bashrc_credentials()
        if base!=config['expected_boyue_base_url']:raise ValueError('Expected Boyue endpoint')
        runtime=Runtime(config,base,key)
    if args.dedup_only:
        from shape_integration.batch import run_dedup
        result=run_dedup(ROOT,ROOT/'runs'/args.run_name,config,items,runtime,
                         source=(ROOT/args.from_localization).resolve(),prepare_only=args.prepare_only)
        return int(any(c['state']=='failed' for c in result['cases']))
    if config['fitting_enabled']:
        if args.from_fitting is not None:
            from shape_integration.recheck import run as recheck
            result=recheck(ROOT,ROOT/'runs'/args.run_name,config,items,runtime,
                           source=(ROOT/args.from_fitting).resolve(),prepare_only=args.prepare_only)
            return int(any(c['state']=='failed' for c in result['cases']))
        from shape_integration.pipeline import run as run_pipeline
        source=args.from_localization
        if source is not None:source=(ROOT/source).resolve()
        result=run_pipeline(ROOT,ROOT/'runs'/args.run_name,config,items,runtime,localization_source=source,prepare_only=args.prepare_only)
        return int(any(c['state']=='failed' for c in result['cases']))
    if args.prepare_only:raise ValueError('Prepare-only requires --from-localization')
    run(ROOT,ROOT/'runs'/args.run_name,config,items,runtime)

if __name__=='__main__':raise SystemExit(main())
