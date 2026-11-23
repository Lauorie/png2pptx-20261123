"""Read-only failed-run evidence reuse into a fresh, independently versioned run."""
from pathlib import Path
import shutil

from ..common import read,save,sha


def seed(prior,output,image):
    prior=Path(prior).resolve();output=Path(output).resolve()
    if prior==output:raise ValueError('Recovery source must be read-only in another directory')
    parameters=read(prior/'provenance.json')['parameters']
    if parameters.get('image_sha256')!=sha(image):raise ValueError('Recovery image mismatch')
    if parameters.get('recognition_mode')!='staged':raise ValueError('Recovery requires staged evidence')
    audit=output/'recovery_seed.json'
    if audit.exists():
        previous=read(audit)
        if previous['source']!=str(prior):raise ValueError('Recovery source changed')
        for name,digest in previous['files'].items():
            if sha(prior/name)!=digest:raise ValueError('Recovery evidence changed: '+name)
        return
    # No partial baseline, native assets, old provenance or scores are copied.
    # Failed global responses remain intact and are revalidated by object.
    names=('recognition.json','crop_recognition.json','staged','glm_crop_geometry','low_score_crop')
    files={}
    for name in names:
        src=prior/name;dest=output/name
        if not src.exists():continue
        if dest.exists():raise ValueError('Refusing to overwrite recovery evidence: '+name)
        paths=list(src.rglob('*')) if src.is_dir() else [src]
        for f in paths:
            if f.is_file():files[str(f.relative_to(prior))]=sha(f)
        if src.is_dir():shutil.copytree(src,dest)
        else:shutil.copy2(src,dest)
    save(audit,{'source':str(prior),'files':files,'policy':'reuse recognition evidence; recompute all layouts/renders'})
