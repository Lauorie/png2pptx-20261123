#!/usr/bin/env python3
"""Render one synthetic native PPT with both bundled LibreOffice runtimes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import fitz
from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from full_recovery.fonts import configure


def main():
    with tempfile.TemporaryDirectory(prefix="png2pptx-native-render-") as directory:
        tmp=Path(directory)
        portable=tmp/"portable"
        shutil.copytree(ROOT/".runtime/portable_lo",portable)
        for path in portable.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.stat().st_size < 4*1024*1024:
                if path.suffix in (".rc",".rdb",".xml",".xcd",".xcu",".ini",".conf") or path.name.endswith("rc") or path.name=="libreoffice":
                    data=path.read_bytes()
                    changed=data.replace(b"{PNG2PPTX_PORTABLE_LO}",str(portable).encode())
                    if changed!=data:path.write_bytes(changed)
        fonts=configure(ROOT/".runtime/text_font_bundle_20261021",portable)
        ppt=Presentation()
        slide=ppt.slides.add_slide(ppt.slide_layouts[6])
        slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(1), Inches(1), Inches(3), Inches(1))
        text=slide.shapes.add_textbox(Inches(1),Inches(3),Inches(5),Inches(1))
        run=text.text_frame.paragraphs[0].add_run()
        run.text="PNG to editable PPTX"
        run.font.name="Times New Roman"
        run.font.size=Pt(24)
        source=tmp/"native.pptx"
        ppt.save(source)
        env={**os.environ,"PPT_PORTABLE_LO_ROOT":str(portable),
             "PPT_CPU_GLIBC_DIR":str(ROOT/".runtime/renderer_support/glibc235"),
             "FONTCONFIG_FILE":str(portable/"fonts.conf")}
        commands={
            "text":[sys.executable,str(ROOT/".runtime/renderer_support/lo_cpu.py")],
            "arrow":[str(ROOT/".runtime/arrow_renderer/soffice")]}
        result={}
        for name,prefix in commands.items():
            out=tmp/name;out.mkdir()
            subprocess.run([*prefix,"--headless","-env:UserInstallation="+(out/"profile").as_uri(),
                "--convert-to","pdf","--outdir",str(out),str(source)],
                env=env,text=True,capture_output=True,check=True,timeout=90)
            pdf=out/"native.pdf"
            with fitz.open(pdf) as document:
                assert len(document)==1
                page=document[0]
                assert "editable PPTX" in page.get_text()
                pix=page.get_pixmap()
                assert len(set(pix.samples))>1
                result[name]={"pdf_bytes":pdf.stat().st_size,"pages":1,"native_text_preserved":True}
        print(json.dumps({"passed":True,"renderers":result,"font_families":list(fonts),
            "scope":"synthetic native PPT only; no recovered PNG or GPU inference"},indent=2))


if __name__=="__main__":
    main()
