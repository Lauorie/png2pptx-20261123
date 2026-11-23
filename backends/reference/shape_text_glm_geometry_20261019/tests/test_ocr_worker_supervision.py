import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def module():
    root=Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('worker',root/'ocr_group_worker.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_resolution_fallback_is_bounded_and_never_increases():
    m=module()
    assert m.attempt_sides(1536)==[1536,1024,768]
    assert m.attempt_sides(768)==[768]


def test_native_crash_marked_before_restart(tmp_path,monkeypatch):
    m=module();calls=[]
    def launch(command):
        calls.append(command)
        if len(calls)==1:
            m.save(tmp_path/'ocr_worker_current.json',{'output':str(tmp_path/'group/attempt_1536'),'pid':123})
            return SimpleNamespace(returncode=-11)
        status=json.loads((tmp_path/'group/attempt_1536/attempt_status.json').read_text())
        assert status['state']=='failed' and status['returncode']==-11
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',launch)
    m.supervise(tmp_path/'manifest.json',{'jobs':[{}]})
    assert len(calls)==2
