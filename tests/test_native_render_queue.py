"""Waiting for the shared renderer is outside its execution deadline."""
import fcntl
import threading
from unittest.mock import patch
from shape_integration.styling.export import render_native


def test_shared_slot_is_acquired_before_the_render_deadline(tmp_path,monkeypatch):
    root=tmp_path/'locks';root.mkdir();held=[]
    for name in ('0','1'):
        handle=(root/name).open('a');fcntl.flock(handle,fcntl.LOCK_EX);held.append(handle)
    monkeypatch.setenv('PPT_RENDER_LOCK_DIR',str(root))
    monkeypatch.setenv('PPT_LIBREOFFICE_RAW','native-renderer')
    started=threading.Event();errors=[]
    class RenderStarted(Exception):pass
    def run(argv,**kwargs):
        assert argv[0]=='native-renderer'
        assert kwargs['timeout']==.01
        assert 'PPT_RENDER_LOCK_DIR' not in kwargs['env']
        started.set();raise RenderStarted()
    def render():
        try:render_native(tmp_path/'input.pptx',tmp_path/'out',[100,100],.01)
        except BaseException as exc:errors.append(exc)
    with patch('shape_integration.styling.export.subprocess.run',run):
        worker=threading.Thread(target=render);worker.start()
        try:
            assert not started.wait(.05)
            held.pop().close()
            worker.join(2)
            assert started.is_set()
            assert len(errors)==1 and isinstance(errors[0],RenderStarted)
        finally:
            for handle in held:handle.close()
            worker.join(2)
