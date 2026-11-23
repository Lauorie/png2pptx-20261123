"""Replay occurrence/font/ink-box fixes on saved text; no API or OCR requests."""
import argparse
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha,compare
from shape_integration.texts import options,layout,export,render_feedback as feedback
from shape_integration.texts.occurrence_dedup import deduplicate
from shape_integration.texts.bbox_color_review import retain_nonregressing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if read(a.old_run/'provenance.json')['parameters']['image_sha256']!=sha(a.image):
        raise ValueError('Source image mismatch')
    a.output.mkdir(parents=True)
    opt=options(read(ROOT/'config.json'));opt.update(pixel_reflow_enabled=False)
    slots=threading.BoundedSemaphore(1)
    original=read(a.old_run/'final/scene.json')['texts']
    rows,audit=deduplicate(original,opt['duplicate_iou'])
    save(a.output/'deduplication.json',audit)
    save(a.output/'provenance.json',{'api_calls':0,'old_run':str(a.old_run),'image_sha256':sha(a.image),
        'source_scene_sha256':sha(a.old_run/'final/scene.json'),
        'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'src').rglob('*.py')}})
    source=Image.open(a.image).convert('RGB')
    base=a.output/'deduplicated_baseline'
    export.final(rows,a.image,base,slots,opt)
    before=Image.open(base/'native.png').convert('RGB')
    print('deduplicated',len(original),'->',len(rows),flush=True)
    # Formula geometry is out of scope; preserve its existing native math and
    # measurements instead of rebuilding equation previews during a text fix.
    fitted,errors=layout.fit_all([r for r in rows if r.get('kind')!='formula'],
                                a.output/'typography',slots,opt,source=a.image)
    by_id={r['id']:r for r in rows}
    updates={r['id']:[r] for r in fitted if not r['layout']['measurement'].get('preview_degraded')}
    conflicts=feedback.new_collisions(rows,feedback.replace(rows,updates))
    blocked={i for pair in conflicts for i in pair}
    updates={i:v for i,v in updates.items() if i not in blocked}
    rounds=[];state={}
    def evaluate(candidates):
        folder=a.output/'cpu_guard'/f'round_{len(rounds)+1:02d}'
        selected=feedback.replace(rows,candidates)
        export.final(selected,a.image,folder,slots,opt)
        actual=Image.open(folder/'native.png').convert('RGB')
        scores={i:feedback.evidence(source,before,actual,[by_id[i]],rs) for i,rs in candidates.items()}
        save(folder/'scene.json',{'texts':selected,'source_size':list(source.size)})
        save(folder/'scores.json',scores);rounds.append(folder);state.update(folder=folder,rows=selected)
        print('CPU gate',len(rounds),'candidates',len(candidates),flush=True)
        return scores
    accepted,history=retain_nonregressing(updates,evaluate,0.)
    import shutil
    shutil.copytree(state['folder'],a.output/'final')
    compare([a.image,a.old_run/'final/native.png',a.output/'final/native.png'],
            ['Original','Previous GLM','Monospace + dedup + ink bounds'],a.output/'comparison.png',width_cap=2100)
    save(a.output/'summary.json',{'before':len(original),'after':len(rows),'deduplicated':audit,
         'accepted_geometry_font_ids':list(accepted),'errors':errors,'api_calls':0,
         'metric':'same-ROI foreground proxy; not ground-truth fidelity','cpu_gate':history})
    print('completed',a.output,flush=True)


if __name__=='__main__':main()
