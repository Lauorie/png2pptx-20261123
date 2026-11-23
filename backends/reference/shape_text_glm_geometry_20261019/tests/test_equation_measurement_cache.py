import subprocess
import fitz
import pytest
from shape_integration.texts import equations


def document(path):
    equations.write_measurement_docx([{'id':'one','latex':'x_1','color':'#000000'}],path)


def pdf(path,pages=1,empty=False):
    result=fitz.open()
    for _ in range(pages):
        page=result.new_page()
        if not empty:page.insert_text((30,30),'x1')
    result.save(path);result.close()


def test_only_complete_matching_nonempty_pdf_is_reusable(tmp_path):
    doc=tmp_path/'math.docx';target=tmp_path/'math.pdf';document(doc);pdf(target)
    assert equations.complete_pdf(doc,target)
    truncated=tmp_path/'truncated.pdf';truncated.write_bytes(target.read_bytes()[:-20])
    assert not equations.complete_pdf(doc,truncated)
    wrong=tmp_path/'wrong.pdf';pdf(wrong,pages=2)
    assert not equations.complete_pdf(doc,wrong)
    blank=tmp_path/'blank.pdf';pdf(blank,empty=True)
    assert not equations.complete_pdf(doc,blank)


def test_finished_pdf_survives_only_exit_timeout(tmp_path,monkeypatch):
    doc=tmp_path/'math.docx';folder=tmp_path/'render';folder.mkdir();document(doc)
    target=folder/'math.pdf';signals=[]
    class Process:
        pid=123456;returncode=-9;attempt=0
        def communicate(self,timeout=None):
            self.attempt+=1
            if self.attempt==1:
                pdf(target)
                raise subprocess.TimeoutExpired('test',1)
            return 'converted',''
    monkeypatch.setattr(equations.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(equations.os,'killpg',lambda pid,sig:signals.append(pid))
    result=equations.render_docx(doc,folder,1)
    assert result==target and signals==[123456]
    assert (folder/'completed_pdf_recovery.json').exists()


def test_exit_timeout_cannot_reuse_stale_pdf(tmp_path,monkeypatch):
    doc=tmp_path/'math.docx';folder=tmp_path/'render';folder.mkdir();document(doc)
    pdf(folder/'math.pdf')
    class Process:
        pid=123456;returncode=-9;attempt=0
        def communicate(self,timeout=None):
            self.attempt+=1
            if self.attempt==1:raise subprocess.TimeoutExpired('test',1)
            return '',''
    monkeypatch.setattr(equations.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(equations.os,'killpg',lambda *a:None)
    with pytest.raises(RuntimeError,match='stale or incomplete'):
        equations.render_docx(doc,folder,1)
