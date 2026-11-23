"""Replay a returned JSON after fixing off-crop existing-point validation.
No model or SAM calls. No coordinates, masks or model decisions are rewritten.
"""
import argparse,copy,json,os,shutil,sys,threading
from pathlib import Path
from PIL import Image
import shared_repair,pipeline
from transport.common import read_json as read,write_json as write,sha256 as sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    if not a.frozen:
        source=root/'runs/shared_repair5_20261018_v3/refocus/cases/ds004';baseline=root/'runs/all40_corner_scope_20261017_v4/cases/ds004'
        write(run/'scope.json',dict(source=str(source),baseline=str(baseline)));shutil.copy2(root/'config.json',run/'config.json');files={}
        for src in [*root.glob('*.py'),*(root/'lib').glob('*.py'),*(root/'transport').glob('*.py')]:
            rel=src.relative_to(root);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files[str(rel)]=sha(dest)
        write(run/'code_manifest.json',dict(files=files,config_sha256=sha(run/'config.json')))
        os.execv(sys.executable,[sys.executable,str(run/'code'/Path(__file__).name),'--run',str(run),'--frozen'])
    scope=read(run/'scope.json');source=Path(scope['source']);baseline=Path(scope['baseline']);case=run/'cases/ds004';case.mkdir(parents=True,exist_ok=True)
    original=source/'continuation/local';dest=case/'continuation/local';dest.mkdir(parents=True,exist_ok=True)
    for name in ['input','sam','masks']:
        if (original/name).exists():shutil.copytree(original/name,dest/name,dirs_exist_ok=True)
    shutil.copy2(baseline/'clean.png',case/'clean.png');shutil.copytree(baseline/'result',case/'before',dirs_exist_ok=True)
    answer=read(original/'request/parsed_0.json');old_prompt=read(original/'prompt.json');evidence=[]
    class Replay:
        def ask(self,folder,prompt,images,validator):
            assert prompt['existing']==old_prompt['existing']
            assert prompt['mask_measurements']==old_prompt['mask_measurements']
            for label,path in images:
                src=original/'input'/path.name
                assert sha(src)==sha(path),(src,path)
                evidence.append(dict(label=label,path=str(path),sha256=sha(path)))
            checked=validator(copy.deepcopy(answer));assert checked==answer
            write(folder/'offline_reuse.json',dict(source=str(original/'request/parsed_0.json'),source_sha256=sha(original/'request/parsed_0.json'),api_calls=0,changed_model_decisions=False))
            return checked
    config=read(run/'config.json');clean=Image.open(case/'clean.png').convert('RGB');task=dict(bbox=read(source/'locator_provenance.json')['bbox_global'],description=old_prompt['focus']['description'])
    rows=read(baseline/'scene.json')['arrows'];locks={k:threading.Semaphore(1) for k in ['cpu','sam']}
    after,patch=shared_repair.recover(clean,task,rows,'P1',case/'continuation',config,Replay(),locks)
    result=shared_repair.merge_results([dict(group='preserved',state='completed',rows=rows)],patch);pipeline.publish(clean,result,case)
    write(case/'completed.json',dict(state=patch['state'],api_calls=0,sam_calls=0,rows=len(patch['rows']),replace=patch['replace']))
    write(run/'completed.json',dict(api_calls=0,sam_calls=0,decisions_unchanged=True,inputs_match_original=True,case='ds004',images=evidence))
    (run/'RESULTS.md').write_text('# ds004：离线修复已有点引用校验\n\n第一次模型输出的所有坐标、头型、替换 ID 均原样保留。原校验器拒绝的两个点，正是输入清单提供的原路径端点。仅允许精确复用被替换路径的既有点；新点仍须在 CLEAN 内。未新增 GLM 或 SAM 请求。\n\n| 原图 | 修复前 | 修复后 |\n|---|---|---|\n| ![](cases/ds004/clean.png) | ![](cases/ds004/before/overlap.png) | ![](cases/ds004/result/overlap.png) |\n\n[局部输入与补丁](cases/ds004/continuation/README.md) · [验证记录](completed.json)\n')
    print(json.dumps(dict(output=str(run),api_calls=0,sam_calls=0,added=len(patch['rows'])),ensure_ascii=False))


if __name__=='__main__':main()
