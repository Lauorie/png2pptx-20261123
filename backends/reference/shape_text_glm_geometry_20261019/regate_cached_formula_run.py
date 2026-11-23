"""Rejudge exact cached formula renders; never synthesize an unrendered mixture."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha,compare
from shape_integration.texts import render_feedback as feedback
from shape_integration.texts.bbox_color_review import DEFAULT_MAX_SCORE_DROP,retain_nonregressing


def comparable_rows(rows):
    # Export refreshes these explanatory fields even for untouched objects.
    return [{k:v for k,v in row.items() if k not in ('background_selection','background_evidence')}
            for row in rows]


def find_exact_render(replay, rows, target_ids):
    for folder in sorted((replay/'cpu_guard').glob('round_*')):
        if not (folder/'scene.json').is_file() or not (folder/'scores.json').is_file():continue
        if set(read(folder/'scores.json'))!=set(target_ids):continue
        if comparable_rows(read(folder/'scene.json')['texts'])==comparable_rows(rows):
            for name in ('text.pptx','native.png','native.pdf'):
                if not (folder/name).is_file():raise ValueError('Cached rendered artifact missing: '+name)
            return folder
    raise ValueError('This candidate combination has no exact saved render; run a fresh formula replay')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','image','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-score-drop',type=float,default=DEFAULT_MAX_SCORE_DROP,
                        help='Absolute 0–1 score tolerance; .05 means five points')
    args=parser.parse_args()
    if not 0<=args.max_score_drop<=1:raise ValueError('Invalid score tolerance')
    if args.output.exists():raise ValueError('Use a fresh output directory')
    parameters=read(args.replay/'replay_parameters.json')
    if parameters['image_sha256']!=sha(args.image):raise ValueError('Source image mismatch')
    baseline=Path(parameters['source_run'])
    baseline_rows=read(baseline/'final/scene.json')['texts']
    by_id={r['id']:r for r in baseline_rows}
    first=args.replay/'cpu_guard/round_01'
    first_rows=read(first/'scene.json')['texts']
    ids=set(read(first/'scores.json'))
    if [r['id'] for r in first_rows]!=[r['id'] for r in baseline_rows]:
        raise ValueError('Formula replay changed object identities/order')
    for row in first_rows:
        original=by_id[row['id']]
        if row['id'] in ids:
            if row.get('kind')!='formula' or original.get('kind')!='formula':
                raise ValueError('Only formula-layout replays are supported')
            for key in ('bbox_2d','latex','text','runs','rotation_deg','color','bold','italic','font'):
                if row.get(key)!=original.get(key):raise ValueError('Cached formula semantics/style changed: '+key)
        elif comparable_rows([row])!=comparable_rows([original]):
            raise ValueError('Cached replay changed a non-target object')
    updates={r['id']:[deepcopy(r)] for r in first_rows if r['id'] in ids}
    source=Image.open(args.image).convert('RGB')
    before=Image.open(baseline/'final/native.png').convert('RGB')
    rounds=[]
    def evaluate(chosen):
        rows=feedback.replace(baseline_rows,chosen)
        folder=find_exact_render(args.replay,rows,chosen)
        actual=Image.open(folder/'native.png').convert('RGB')
        saved=read(folder/'scores.json')
        metrics={i:feedback.evidence(source,before,actual,[by_id[i]],parts) for i,parts in chosen.items()}
        for ident,metric in metrics.items():
            for side in ('before','after'):
                if abs(metric[side]['score']-saved[ident][side]['score'])>1e-8:
                    raise ValueError('Cached source/render score no longer matches: '+ident)
        rounds.append(folder)
        return metrics
    accepted,history=retain_nonregressing(updates,evaluate,args.max_score_drop)
    args.output.mkdir(parents=True)
    shutil.copytree(rounds[-1],args.output/'final',ignore=shutil.ignore_patterns('profile'))
    save(args.output/'cpu_guard/history.json',history)
    save(args.output/'replay_parameters.json',{**parameters,'max_score_drop':args.max_score_drop})
    save(args.output/'provenance.json',{'parameters':{
        'image_sha256':sha(args.image),'source_run':str(baseline),
        'source_replay':str(args.replay.resolve()),'max_score_drop':args.max_score_drop,
        'policy':'exact_cached_formula_render_regating'},
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                       [Path(__file__),ROOT/'src/shape_integration/texts/bbox_color_review.py']},
        'reused_renders':[{'path':str(p.resolve()),'pptx_sha256':sha(p/'text.pptx'),
                          'scene_sha256':sha(p/'scene.json'),'native_png_sha256':sha(p/'native.png')}
                         for p in rounds]})
    old=read(args.replay/'summary.json')
    summary={**old,'accepted_ids':sorted(accepted),'reverted_ids':sorted(ids-set(accepted)),
        'max_score_drop':args.max_score_drop,'new_api_calls':0,'new_cpu_renders':0,
        'reused_render':str(rounds[-1].resolve()),'quality_status':'needs_review'}
    save(args.output/'summary.json',summary)
    save(args.output/'usage.json',{'new_api_calls':0,'new_cpu_renders':0})
    compare([args.image,args.replay/'final/native.png',args.output/'final/native.png'],
            ['Original','Previous gate','Five-point tolerance; exact cached render'],
            args.output/'comparison.png',width_cap=1800)
    print(summary,flush=True)


if __name__=='__main__':main()
