"""Run the frozen staged text backend with local transport and native adapters.

Future backend freezes record the source occurrence recovery policy separately
in source_provenance.json; active and historical freezes stay unchanged.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def composition_rows(scene, plan):
    """Remove registered duplicates without changing the supplied typography."""
    dropped = set(plan.get('drop_text_ids', ()))
    return [row for row in scene['texts'] if row['id'] not in dropped]


def load_backend(root):
    sys.path[:0]=[str(root/'.deps'),str(root/'src'),str(root/'vendor'),str(root)]
    from full_recovery.text_breaks import install
    install()
    from shape_integration import common
    from full_recovery.request_reuse import install as install_request_reuse
    install_request_reuse(common)
    original=common.ExperimentDetectorClient
    class LocalDetector(original):
        def request_parameters(self):
            parameters=super().request_parameters()
            parameters.pop('thinking',None)
            effort=os.environ.get('PPT_FORCE_REASONING_EFFORT') or parameters.pop('reasoning_effort',None)
            parameters.pop('reasoning_effort',None)
            parameters.setdefault('chat_template_kwargs',{})['clear_thinking']=True
            if effort is not None:parameters['chat_template_kwargs']['reasoning_effort']=effort
            return parameters
    common.ExperimentDetectorClient=LocalDetector
    # Keep the reference's original-source attribute prompt and validator.
    # Replacing them with an A/B/C orientation board made rotated words look
    # like stacked text, shrinking native type to fit a tall unrotated frame.
    from full_recovery.text_inventory import install as install_inventory_review
    install_inventory_review()
    # Some models drop the "text" key of a run; restore it instead of failing the image.
    from full_recovery.text_json_repair import install as install_json_repair
    install_json_repair()
    # The audit regions and crop groups do not depend on each other: ask them together.
    from full_recovery.text_parallel import install as install_parallel_requests
    install_parallel_requests()
    from full_recovery.source_text_attributes import install as install_source_attributes
    install_source_attributes()
    from full_recovery.text_spacing import install as install_text_spacing
    install_text_spacing()
    from full_recovery.text_angle_transport import install as install_angle_transport
    install_angle_transport()
    # Bold is judged against the rest of the figure; a magnified crop cannot overrule it.
    from full_recovery.text_weight import install as install_text_weight
    install_text_weight()
    from full_recovery.formula_text import install as install_formula_text
    install_formula_text()
    # LO's disposable DOCX preview interprets a literal <n> superscript as an
    # incomplete comparison. Quote these glyphs only in the preview tree; native
    # PowerPoint OMML remains unchanged, like the existing '='/'*' compatibility.
    from shape_integration.texts import equations
    import re
    word_math=equations.word_math
    def compatible_word_math(*args,**kwargs):
        tree=word_math(*args,**kwargs)
        for script in tree.findall('.//m:sup',equations.NS):
            value=''.join(script.xpath('.//m:t/text()',namespaces=equations.NS)).strip()
            if re.fullmatch(r'<[^<>]+>',value):
                for run in script.findall('.//m:r',equations.NS):
                    t=run.find('m:t',equations.NS)
                    if t is not None and t.text in ('<','>'):
                        prop=run.find('m:rPr',equations.NS)
                        if prop is None:
                            prop=equations.element(equations.OMML_NS,'rPr');run.insert(0,prop)
                        if prop.find('m:nor',equations.NS) is None:prop.append(equations.element(equations.OMML_NS,'nor'))
        return tree
    equations.word_math=compatible_word_math
    from shape_integration.texts import equation_recovery
    from full_recovery.math_alphabets import preview_row
    runs_preview=equation_recovery.runs_preview
    def portable_runs_preview(row,folder):
        preview,changes=preview_row(row)
        result=runs_preview(preview,folder)
        if changes:result['math_alphabet_fallback']=changes
        return result
    equation_recovery.runs_preview=portable_runs_preview
    from shape_integration.texts import export
    from full_recovery.text_bounds import normalize_native_text_frames
    original_write=export.write
    def write_native_frames(rows,size,path,base=None):
        result=original_write(rows,size,path,base)
        result['native_frame_normalizations']=normalize_native_text_frames(path,size,texts=rows)
        return result
    export.write=write_native_frames
    return common


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',type=Path,required=True)
    p.add_argument('--render-pptx',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--paint-on-base',type=Path);p.add_argument('--text-scene',type=Path)
    p.add_argument('--source-image',type=Path);p.add_argument('--ownership-plan',type=Path)
    p.add_argument('--shape-scene',type=Path);p.add_argument('--shape-styles',type=Path)
    p.add_argument('--arrow-scene',type=Path)
    p.add_argument('--reuse-output-root',type=Path,help='Run directory whose request folders mirror the reference runs')
    p.add_argument('--reuse-request-root',type=Path,action='append',default=[],help='Finished run whose identical requests are answered from its replies')
    p.add_argument('--refit-formulas',action='store_true')
    p.add_argument('--repair-formula-text',action='store_true')
    p.add_argument('--repair-text-occurrences',action='store_true')
    p.add_argument('--size',nargs=2,type=int);p.add_argument('--test',action='store_true')
    args,rest=p.parse_known_args();root=args.backend.resolve();load_backend(root)
    if args.repair_text_occurrences:
        import shutil
        from shape_integration.common import read,save,sha,Runtime
        from shape_integration.texts.layout import fit
        from shape_integration.texts.export import write,retained_images
        from full_recovery.text_occurrences import ground,reconcile_recalls,refit_position
        from full_recovery.text_product import remove_plain_text
        from full_recovery.text_occlusion import select,write_backings
        from PIL import Image
        if not all((args.text_scene,args.source_image,args.output)):
            p.error('Occurrence repair needs text scene, source PNG and output')
        args.output.mkdir(parents=True,exist_ok=True);scene=read(args.text_scene)
        text_pptx=args.text_scene.with_name('text.pptx')
        inputs=[args.text_scene,text_pptx,args.source_image]
        if args.paint_on_base:inputs.append(args.paint_on_base)
        original_hashes={str(path):sha(path)for path in inputs}
        with Image.open(args.source_image)as image:source=image.convert('RGB')
        if scene['source_size']!=list(source.size):raise ValueError('Text source coordinate grid differs')
        rows,changed,audit=ground(scene['texts'],source)
        config=read(root/'config.json')
        rows=[refit_position(row,fit)if row['id']in changed else row for row in rows]
        rows,aliases=reconcile_recalls(rows,source);dropped={r['id']for r in aliases}
        if changed and args.paint_on_base:
            if not all((args.shape_scene,args.shape_styles,args.arrow_scene,args.ownership_plan)):
                p.error('Changed occurrence backings need their existing shape/arrow ownership inputs')
            runtime=Runtime({**config,'api_concurrency':4,'shape_schema_retries':1,
                'reasoning_effort':'low','request_timeout_seconds':360},config['expected_boyue_base_url'],'')
            old={r['id']:r for r in rows}
            selected,fill_audit=select(rows,read(args.shape_scene),read(args.shape_styles),read(args.arrow_scene),
                read(args.ownership_plan),args.source_image,args.output/'backing_review',runtime,save,
                retained_images(args.paint_on_base,scene['source_size']),review_ids=set(changed))
            rows=[r if r['id']in changed else old[r['id']]for r in selected]
        else:
            if changed and any(r.get('background_fill') or r.get('background_box_px')for r in rows):
                p.error('Painted occurrence repair needs its existing non-text base to review moved backings')
            fill_audit={'model_calls':0,'unpainted_text_product':not bool(args.paint_on_base)}
        if changed or dropped:
            remove_plain_text(text_pptx,args.output/'text.pptx',set(changed)|dropped)
            write([r for r in rows if r['id']in changed],scene['source_size'],args.output/'text.pptx',base=args.output/'text.pptx')
            write_backings(args.output/'text.pptx',rows,scene['source_size'])
        else:shutil.copy2(text_pptx,args.output/'text.pptx')
        save(args.output/'scene.json',{**scene,'texts':rows})
        save(args.output/'repair.json',{'protocol':'source_grounded_text_occurrences.1','changed_ids':changed,
             'retired_recalls':aliases,'observations':audit,'backings':fill_audit,
             'source_files_sha256':original_hashes,'new_text_recognition_stages':0,'shape_arrow_geometry_edited':False,
             'retained_text_font_size_style_angle_unchanged':True})
        if any(sha(path)!=digest for path,digest in original_hashes.items()):
            raise ValueError('Input changed during occurrence repair')
        return
    if args.refit_formulas or args.repair_formula_text:
        # Rendering-environment repair only. Recognition, source bboxes and
        # plain text remain unchanged; measure the existing equations again.
        import threading
        from shape_integration.common import read,save,sha
        from shape_integration.texts import options
        from shape_integration.texts.equation_recovery import measure_resilient
        from shape_integration.texts.layout import fit
        from shape_integration.texts.export import write
        from shape_integration.texts.equations import render_preview
        if not all((args.text_scene,args.source_image,args.output)):
            p.error('Formula refit requires text scene, source image and output')
        scene=read(args.text_scene);rows=scene['texts'];formulas=[r for r in rows if r.get('kind')=='formula']
        if args.repair_formula_text:
            from shape_integration.texts.formula_state import adopt
            import re
            selected={r['id'] for r in formulas if re.search(r'\\texttt\s*\{',r.get('latex',''))}
            rows=[adopt(r,{'latex':r['latex'],'reason':'Balanced text-mode conversion repair'})
                  if r['id'] in selected else r for r in rows]
            formulas=[r for r in rows if r['id'] in selected]
        if scene.get('source_sha256') and scene['source_sha256']!=sha(args.source_image):
            raise ValueError('Formula source differs from the text scene')
        args.output.mkdir(parents=True,exist_ok=True);slots=threading.Semaphore(1)
        opt=options(read(root/'config.json'))
        measured,audit=measure_resilient(formulas,args.output/'equations',slots,opt['render_timeout_seconds']) if formulas else ({},[])
        rows=[fit(r,measured[r['id']],opt['font_fit_fraction']) if r['id'] in measured else r for r in rows]
        write(rows,scene['source_size'],args.output/'text.pptx')
        render_preview(args.output/'text.pptx',args.output/'native_render',scene['source_size'],opt['render_timeout_seconds'])
        save(args.output/'scene.json',{**scene,'texts':rows,'formula_measurement_repair':{
            'source_scene':str(args.text_scene),'source_scene_sha256':sha(args.text_scene),
            'formula_ids':list(measured),'measurement_audit':audit,'model_calls':0,
            'balanced_text_conversion_repair':args.repair_formula_text,
            'recognition_and_source_boxes_unchanged':True}})
        return
    if args.test:
        import pytest
        raise SystemExit(pytest.main([str(root/'tests'),'-q','--disable-warnings',
            '--basetemp',str(root.parents[1]/'tests_tmp/text')]))
    if args.render_pptx:
        from shape_integration.texts.equations import render_preview
        render_preview(args.render_pptx,args.output/'render',args.size,180)
        return
    if args.paint_on_base:
        from shape_integration.common import read,save,sha,Runtime
        from shape_integration.texts.export import retained_images
        from full_recovery.text_product import select_native_text
        from full_recovery.text_occlusion import select,write_backings
        if not all((args.shape_scene,args.shape_styles,args.arrow_scene,args.source_image)):
            p.error('Composition paint requires original image and shape/style/arrow scenes')
        scene=read(args.text_scene);plan=read(args.ownership_plan)
        rows=composition_rows(scene,plan)
        args.output.mkdir(parents=True,exist_ok=True)
        config=read(root/'config.json')
        # Retry an incomplete/invalid response once; valid visual decisions stay cached.
        config.update(api_concurrency=8,shape_schema_retries=1,reasoning_effort='low',request_timeout_seconds=360)
        if args.reuse_output_root and args.reuse_request_root:
            config.update(reuse_output_root=str(args.reuse_output_root),reuse_request_roots=[str(r) for r in args.reuse_request_root])
        runtime=Runtime(config,config['expected_boyue_base_url'],'')
        from PIL import Image
        from full_recovery.text_coverage import candidates as coverage_candidates, observe as observe_text_coverage
        from shape_integration.styling.export import render_native
        from shape_integration.texts.layout import fit_all
        from shape_integration.texts import options as text_options
        import threading
        with Image.open(args.source_image) as opened:
            clean=opened.convert('RGB')
        from full_recovery.text_occurrences import ground as ground_occurrences, reconcile_recalls,refit_position
        rows,grounded_ids,grounding=ground_occurrences(rows,clean)
        from shape_integration.texts.layout import fit
        rows=[refit_position(row,fit)
              if row['id'] in grounded_ids else row for row in rows]
        rows,grounded_aliases=reconcile_recalls(rows,clean)
        save(args.output/'source_occurrence_grounding.json',{'observations':grounding,
             'changed_ids':grounded_ids,'retired_recalls':grounded_aliases,'extra_model_stages':0})
        extra=[];marks=[];coverage={'model_calls':0,'added_text':[]};native=None;region_rows=[];region_audit={'regions':0}
        if coverage_candidates(clean, rows):
            preview=render_native(args.paint_on_base,args.output/'source_coverage/base/native_render',scene['source_size'],180)
            with Image.open(preview) as opened:
                native=opened.convert('RGB')
            from full_recovery.source_marks import recover as recover_marks
            marks=recover_marks(clean,native,coverage_candidates(clean,rows,native))
            save(args.output/'source_coverage/source_marks.json',marks)
            known=[*rows,*[{'id':m['id'],'bbox_2d':m['bbox_2d'],'font':'Arial'} for m in marks]]
            # Dense lettering nobody recovered: left out by the text branch as picture-internal, not kept as a picture.
            from full_recovery.text_regions import observe as read_unowned_regions
            region_rows,region_audit=read_unowned_regions(clean,coverage_candidates(clean,known,native),known,args.output/'source_regions',runtime,save)
            if region_rows:
                opt=text_options(config);opt['native_font_metrics']=True
                region_rows,errors=fit_all(region_rows,args.output/'source_regions/native_layout',threading.Semaphore(1),opt)
                region_audit['native_layout_errors']=errors
                known=[*known,*region_rows]
            extra,coverage=observe_text_coverage(clean,native,known,args.output/'source_coverage',runtime,save)
            if extra:
                opt=text_options(config);opt['native_font_metrics']=True
                extra,errors=fit_all(extra,args.output/'source_coverage/native_layout',threading.Semaphore(1),opt)
                coverage['native_layout_errors']=errors
                from shape_integration.texts.export import write as write_probe
                from full_recovery.text_coverage import verify_glyphs
                probe=args.output/'source_coverage/glyph_probe.pptx';write_probe(extra,scene['source_size'],probe)
                rendered=render_native(probe,args.output/'source_coverage/glyphs/native_render',scene['source_size'],180)
                with Image.open(rendered) as opened:
                    extra,verification=verify_glyphs(clean,opened,extra)
                coverage['glyph_verification']=verification
                coverage['added_text']=extra
        extra=[*region_rows,*extra];coverage['unowned_regions']={k:v for k,v in region_audit.items() if k!='added_text'}
        original_ids={r['id'] for r in rows};extra_ids={r['id'] for r in extra}
        rows+=extra
        rows,new_aliases=reconcile_recalls(rows,clean)
        grounded_aliases+=new_aliases
        extra_ids-= {r['id'] for r in new_aliases}
        extra=[r for r in extra if r['id'] in extra_ids]
        from full_recovery.source_punctuation import candidates as punctuation_candidates, observe as native_punctuation
        punctuation_receipts=[];glyph_native=None
        if punctuation_candidates(rows):
            if native is None:
                preview=render_native(args.paint_on_base,args.output/'source_punctuation/base/native_render',scene['source_size'],180)
                with Image.open(preview) as opened:native=opened.convert('RGB')
            glyph_preview=render_native(args.text_scene.with_name('text.pptx'),
                args.output/'source_punctuation/glyph/native_render',scene['source_size'],180)
            with Image.open(glyph_preview) as opened:glyph_native=opened.convert('RGB')
            rows,punctuation_receipts=native_punctuation(clean,native,rows,read(args.shape_scene),
                read(args.shape_styles),read(args.arrow_scene),plan,glyph_native=glyph_native)
            save(args.output/'source_punctuation/native_replacements.json',{
                'native_view':str(preview),'native_view_sha256':sha(preview),
                'glyph_view':str(glyph_preview),'glyph_view_sha256':sha(glyph_preview),
                'source_sha256':sha(args.source_image),'receipts':punctuation_receipts})
        rows,review=select(rows,read(args.shape_scene),read(args.shape_styles),read(args.arrow_scene),plan,
                          args.source_image,args.output/'text_occlusion',runtime,save,
                          retained_images(args.paint_on_base,scene['source_size']),glyph_native=glyph_native)
        # Fix known degraded fallback glyphs without new recognition. Native
        # formula semantics and source boxes stay fixed; other labels retain
        # their supplied native objects and metrics.
        from full_recovery.math_alphabets import preview_row
        from shape_integration.texts import equation_recovery
        from shape_integration.texts.layout import fit
        repaired={row['id']:row for row in rows if row['id'] in grounded_ids};preview_repairs=[]
        for row in rows:
            measurement=row.get('layout',{}).get('measurement',{})
            if (row.get('kind')!='formula' or not measurement.get('preview_degraded')
                    or measurement.get('math_alphabet_fallback') or not preview_row(row)[1]):continue
            metric=equation_recovery.runs_preview(row,args.output/'formula_preview_repair')
            repaired[row['id']]=fit(row,metric,text_options(config)['font_fit_fraction'])
            preview_repairs.append({'text_id':row['id'],'glyph_changes':metric['math_alphabet_fallback'],
                'native_formula_and_source_box_unchanged':True,'preview_remains_explicitly_degraded':True})
        select_native_text(args.text_scene.with_name('text.pptx'),args.output/'text.pptx',
            [r for r in rows if r['id'] in original_ids and r['id'] not in repaired])
        if repaired:
            from shape_integration.texts.export import write as append_repaired
            append_repaired(list(repaired.values()),scene['source_size'],args.output/'text.pptx',base=args.output/'text.pptx')
            rows=[repaired.get(r['id'],r) for r in rows]
        from full_recovery.source_ellipsis import observe as observe_ellipsis, replace as replace_ellipsis
        ellipses=observe_ellipsis(clean,rows,[r for r in read(args.arrow_scene)['arrows']
            if r['id'] not in set(plan.get('drop_arrow_names',[]))])
        if extra:
            from shape_integration.texts.export import write as append_native_text
            append_native_text([r for r in rows if r['id'] in extra_ids],scene['source_size'],args.output/'text.pptx',base=args.output/'text.pptx')
        if marks:
            from full_recovery.source_marks import append as append_marks
            append_marks(args.output/'text.pptx',marks,scene['source_size'])
        # All text occurrences, including coverage additions, exist before
        # their source-dot replacement. Appending an ellipsis afterwards would
        # recreate the glyph beside its measured native triplet.
        replace_ellipsis(args.output/'text.pptx',ellipses,scene['source_size'])
        ellipsis_aliases={ident for mark in ellipses for ident in mark.get('duplicate_text_ids',[])}
        rows=[row for row in rows if row['id'] not in ellipsis_aliases]
        review['native_text_product_preserved']=not bool(repaired)
        review['formula_preview_repairs']=preview_repairs
        review['source_occurrence_grounding']={'changed_ids':grounded_ids,
            'retired_recalls':grounded_aliases,'observations':grounding,'extra_model_stages':0}
        review['source_text_coverage']=coverage
        review['native_source_marks']=marks
        review['source_ellipsis']=ellipses
        review['removed_source_ellipsis_aliases']=sorted(ellipsis_aliases)
        review['native_punctuation_replacements']=punctuation_receipts
        review['removed_native_punctuation_duplicates']=[r['text_id'] for r in punctuation_receipts
            if r['action']=='retire_duplicate_text']
        if repaired or extra or marks or ellipses or review['removed_native_punctuation_duplicates']:
            review['text_geometry_and_typography_unchanged']=False
        write_backings(args.output/'text.pptx',rows,scene['source_size'])
        from full_recovery.text_bounds import clamp_small_text_overflow,normalize_native_text_frames
        review['native_text_frame_normalizations']=normalize_native_text_frames(
            args.output/'text.pptx',scene['source_size'],texts=rows)
        adjustments=clamp_small_text_overflow(args.output/'text.pptx',scene['source_size'],texts=rows)
        for change in adjustments:
            row=next(r for r in rows if r['id']==change['text_id'])
            row['frame_px']=[*change['after_xy_px'],*row['frame_px'][2:]]
        review['text_frame_adjustments']=adjustments
        if adjustments:review['text_geometry_and_typography_unchanged']=False
        save(args.output/'scene.json',{**scene,'texts':rows,'native_source_marks':marks})
        save(args.output/'composition.json',{**review,'removed_duplicate_ids':plan.get('drop_text_ids',[])})
        return
    os.environ['PPT_LOCAL_GLM']='1'
    # Non-secret sentinel for the reference CLI's API guards. Authentication
    # is disabled on the frozen local endpoint.
    os.environ.setdefault('OpenAI_API','local-glm-without-auth')
    # The copied CLI accepts an unauthenticated local endpoint explicitly.
    # Its recognition, geometry, equation and OCR code is otherwise unchanged.
    if args.output is None:p.error('--output is required for text recovery')
    sys.argv=[str(root/'run_global_ocr_review.py'),*rest,'--output',str(args.output)]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
