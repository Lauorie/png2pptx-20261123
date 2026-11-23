"""Freeze and resolve the metric-compatible fonts used by the text reference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from xml.sax.saxutils import escape


def freeze(source, target):
    manifest=json.loads((source/'manifest.json').read_text())
    target.mkdir(parents=True)
    for name,digest in manifest['files'].items():
        path=source/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Font asset changed: '+name)
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    shutil.copytree(source/'licenses',target/'licenses')
    shutil.copy2(source/'manifest.json',target/'manifest.json')


def configure(bundle, portable):
    bundle, portable = Path(bundle).resolve(), Path(portable).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:raise ValueError('Font asset changed: '+name)
    math_dir=Path(__file__).resolve().parent/'assets/math_font'
    math_manifest=json.loads((math_dir/'manifest.json').read_text())
    math_file=math_dir/'DejaVuMathTeXGyre.ttf'
    if hashlib.sha256(math_file.read_bytes()).hexdigest()!=math_manifest['sha256']:
        raise ValueError('Math font asset changed')
    symbol_dir = Path(__file__).resolve().parent / 'assets/text_symbol_fonts'
    symbols = json.loads((symbol_dir / 'manifest.json').read_text())
    for name, digest in symbols['files'].items():
        if hashlib.sha256((symbol_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Symbol font asset changed: ' + name)
    config=portable/'fonts.conf'
    config.write_text('<fontconfig><dir>'+escape(str(bundle/'fonts'))+'</dir><dir>'+escape(str(portable/'fonts'))+
        '</dir><dir>'+escape(str(symbol_dir))+'</dir><dir>'+escape(str(math_dir))+'</dir><cachedir>'+escape(str(portable/'font-cache'))+'</cachedir><include>'+
        escape(str(bundle/'metric-aliases.conf'))+'</include>'+
        '<alias binding="strong"><family>Cambria Math</family><prefer><family>DejaVu Math TeX Gyre</family></prefer></alias>'+
        '<match target="pattern"><test name="family" compare="not_eq" qual="all"><string>Cambria Math</string></test>'+
        '<edit name="family" mode="append" binding="weak"><string>DejaVu Sans</string></edit></match>'+
        '</fontconfig>')
    env={**os.environ,'FONTCONFIG_FILE':str(config)};resolved={}
    for family,expected in manifest['resolved'].items():
        lines=subprocess.check_output(['fc-match','-f','%{family[0]}\n%{file}\n',family],env=env,text=True).splitlines()
        digest=hashlib.sha256(Path(lines[1]).read_bytes()).hexdigest()
        if lines[0]!=expected['resolved'] or digest!=expected['sha256']:
            raise ValueError('Font substitution differs from the frozen reference: '+family)
        resolved[family]={'family':lines[0],'file':lines[1],'sha256':digest}
    lines=subprocess.check_output(['fc-match','-f','%{family[0]}\n%{file}\n','Cambria Math'],env=env,text=True).splitlines()
    if hashlib.sha256(Path(lines[1]).read_bytes()).hexdigest()!=math_manifest['sha256']:
        raise ValueError('Math font fallback is unavailable')
    resolved['Cambria Math']={'family':lines[0],'file':lines[1],'sha256':math_manifest['sha256']}
    lines=subprocess.check_output(['fc-match','-f','%{family[0]}\n%{file}\n','Arial:charset=22ee'],env=env,text=True).splitlines()
    if hashlib.sha256(Path(lines[1]).read_bytes()).hexdigest() != symbols['files']['DejaVuSans.ttf']:
        raise ValueError('Text symbol fallback differs from the supplied native text renderer')
    resolved['text_symbol_fallback']={'family':lines[0],'file':lines[1],'sha256':symbols['files']['DejaVuSans.ttf']}
    return resolved
