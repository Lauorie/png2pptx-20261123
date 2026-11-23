"""Offline follow-up using actual C decisions and unchanged baseline geometry."""
import argparse,copy,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import numpy as np
from PIL import Image
import style_windows
from lib import viz
from transport.common import read_json as read,write_json as write


def main(run,baseline):
    run=Path(run).resolve();base=Path(baseline).resolve();results=[]
    lines=['# 箭身取样后：补足墨段边界的离线复测','',
           '固定 GLM 返回的取样、原有几何和头型。沿取样直线在原图测量完整墨段，只使用与指定取样范围相交的墨段，不采用远处其他段。无新增模型/SAM 请求。',
           '', '| case | 原图 | 旧样式 | 首次取样测量 | 补足墨段边界 |', '|---|---|---|---|---|']
    for folder in sorted((run/'cases').glob('*/*/C')):
        if not (folder/'fixed_rows_local.json').exists():continue
        cid,gid=folder.parent.parent.name,folder.parent.name
        old=base/'cases'/cid/'groups'/gid;rows=read(folder/'fixed_rows_local.json');saved=copy.deepcopy(rows)
        view=read(folder/'input/transform.json');scale=float(np.sqrt(abs(np.linalg.det(np.array(view['view_to_global'])[:2,:2]))))
        image=np.array(Image.open(folder/'input/clean.png').convert('RGB'));checks=[]
        for row in rows:
            if 'shaft_sample' not in row or row['shaft_sample']=='auto':continue
            row['style']=style_windows.recover(image,None,row)
            checks.append(dict(id=row['id'],shaft_sample=row['shaft_sample'],width_global=(row['style']['width_px']*scale if row['style']['width_px'] is not None else None),
                color=row['style']['color'],dash=row['style']['dash'],needs_review=row['style']['needs_review']))
        assert all({k:v for k,v in a.items() if k!='style'}=={k:v for k,v in b.items() if k!='style'} for a,b in zip(rows,saved))
        dest=folder/'fixed_style_v2';write(dest/'rows_local.json',rows)
        if rows:viz.render(folder/'input/clean.png',rows,dest)
        write(dest/'audit.json',dict(model_calls=0,geometry_and_head_enums_unchanged=True,checks=checks))
        if not rows:continue
        paths=[folder/'input/clean.png',old/'result/styled.png',folder/'fixed_style/styled.png',dest/'styled.png']
        lines.append('| '+cid+'/'+gid+' | '+' | '.join('![]('+os.path.relpath(p,run)+')' for p in paths)+' |')
        results.append(dict(case=cid,group=gid,checks=checks))
    (run/'STYLE_REPLAY.md').write_text('\n'.join(lines)+'\n\n具体取样、线宽、周期与未知状态见 [审计](style_replay.json)。ds027 如仍无可靠箭身，则样式保持未解决，不把不显示的大头当作修复成功。\n')
    write(run/'style_replay.json',dict(model_calls=0,results=results))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True)
    a=p.parse_args();main(a.run,a.baseline)
