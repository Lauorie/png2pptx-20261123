"""Full-inventory alternating PPT review; missing results remain visible placeholders."""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt

from ..scene import save
from .matte import checkerboard


def read(path): return json.loads(Path(path).read_text())


def build_report(root, output):
    root = Path(root).resolve(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = read(root/'batch_manifest.json'); prs = Presentation()
    prs.slide_width = Inches(13.333333); prs.slide_height = Inches(8.2)
    pages, rows, records, cases = [], [], [], []
    for case in manifest['cases']:
        case_root = Path(case['run']); scene_path = case_root/'image_stage.json'
        stage = read(scene_path) if scene_path.exists() else {'assets': {}}
        inventory_path = case_root/'inventory.json'
        inventory = read(inventory_path)['assets'] if inventory_path.exists() else case.get('expected_assets', [])
        ids = list(dict.fromkeys([a['id'] for a in case.get('expected_assets', [])]+[a['id'] for a in inventory]))
        asset_hints = {a['id']: a for a in case.get('expected_assets', [])+inventory}
        assets = dict(stage['assets'])
        for ident, result_path in case.get('asset_overrides', {}).items():
            if ident not in ids: raise ValueError('Override is not an inventory member: '+ident)
            assets[ident] = read(result_path)
        ready = [i for i in ids if assets.get(i, {}).get('rgba')]
        case_status = {'case': case['id'], 'stage_status': stage.get('status', 'pending'),
                       'coverage_status': stage.get('coverage_status', 'unknown'),
                       'boundary_review_status': stage.get('boundary_review', {}).get('status', 'not_recorded'),
                       'inventory_available': inventory_path.exists(), 'error': stage.get('error'),
                       'detected_count': len(ids) if inventory_path.exists() else None,
                       'generated_count': len(ready)}
        cases.append(case_status)
        coverage_label = case_status['coverage_status'] if inventory_path.exists() else 'INVENTORY UNAVAILABLE'
        boundary_label = case_status['boundary_review_status']
        with Image.open(case['source']) as source: w, h = source.size
        scale = min((prs.slide_width-Inches(.4))/w, (prs.slide_height-Inches(.9))/h)
        ox = (prs.slide_width-w*scale)/2; oy = Inches(.55)+(prs.slide_height-Inches(.9)-h*scale)/2
        for method in ('original', 'SenseNova + alpha'):
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            fill = slide.background.fill; fill.solid()
            fill.fore_color.rgb = RGBColor(255, 255, 255) if method=='original' else RGBColor(238, 238, 238)
            title = slide.shapes.add_textbox(Inches(.2), Inches(.08), Inches(12.9), Inches(.3)).text_frame.paragraphs[0]
            title.text = f'{case["id"]} | {method}' + (f' | {len(ready)}/{len(ids) if inventory_path.exists() else "?"} generated; search {coverage_label}; bounds {boundary_label}; fidelity unverified' if method!='original' else '')
            title.font.size = Pt(14)
            if method == 'original':
                slide.shapes.add_picture(case['source'], round(ox), round(oy), round(w*scale), round(h*scale))
            else:
                if not ids:
                    text = slide.shapes.add_textbox(Inches(.5), Inches(.7), Inches(12), Inches(.4)).text_frame.paragraphs[0]
                    text.text = ('Detection failed or unfinished; asset count UNKNOWN' if not inventory_path.exists()
                                 else 'No image assets returned by detector; coverage: '+coverage_label)
                    text.font.size = Pt(16)
                for ident in ids:
                    asset = assets.get(ident, {}); hint = asset_hints.get(ident, {})
                    if asset.get('rgba'):
                        x0,y0,x1,y1 = asset['placement_source_half_open']
                        shape = slide.shapes.add_picture(asset['rgba'], round(ox+x0*scale), round(oy+y0*scale),
                                                        max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale)))
                        shape.name = case['id']+'/'+ident+' ['+asset['status']+']'
                    else:
                        b = hint.get('source_box_half_open')
                        if b is None and 'bbox' in hint:
                            iw,ih = read(case_root/'inventory.json')['input_size']
                            b = [(hint['bbox'][i]+.5)*(w/iw if i%2==0 else h/ih) for i in range(4)]
                        if b:
                            shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, round(ox+b[0]*scale), round(oy+b[1]*scale),
                                                          max(1, round((b[2]-b[0])*scale)), max(1, round((b[3]-b[1])*scale)))
                            shape.fill.background(); shape.line.color.rgb = RGBColor(220, 40, 40)
                            shape.text = ident+' '+asset.get('status', 'PENDING')
            pages.append({'slide': len(prs.slides), 'case': case['id'], 'method': method,
                          'inventory_count': len(ids) if inventory_path.exists() else None,
                          'generated_count': len(ready), 'missing_ids': sorted(set(ids)-set(ready)),
                          'coverage_status': coverage_label, 'stage_status': case_status['stage_status']})
            pages[-1]['boundary_review_status'] = boundary_label
        for ident in ids:
            asset = assets.get(ident, {})
            folder = Path(asset['rgba']).parent if asset.get('rgba') else Path(case.get('asset_overrides', {}).get(ident, str(case_root/'assets'/ident/'result.json'))).parent
            record = {'case': case['id'], 'id': ident, 'status': asset.get('status', 'pending'),
                      'error': asset.get('error'), 'quality_flags': asset.get('quality_flags', []),
                      'prompt': str(folder/'prompt.txt'), 'source': str(folder/'source_context.png'),
                      'raw_edit': str(folder/'edit_raw.png'), 'rgba': asset.get('rgba')}
            records.append(record)
            row = Image.new('RGB', (1080, 370), 'white'); draw = ImageDraw.Draw(row)
            draw.text((8, 5), f'{case["id"]}/{ident} | source context / raw edit / alpha | '+record['status'], fill='black')
            for index, filename in enumerate(('source_context.png', 'edit_raw.png', 'rgba.png')):
                path = folder/filename
                if not path.exists():
                    draw.text((index*360+10, 100), 'MISSING: '+record['status'], fill='red'); continue
                with Image.open(path) as im:
                    panel = checkerboard(im) if index==2 else im.convert('RGB')
                factor = min(350/panel.width, 325/panel.height)
                panel = panel.resize((max(1, round(panel.width*factor)), max(1, round(panel.height*factor))), Image.Resampling.LANCZOS)
                row.paste(panel, (index*360+(360-panel.width)//2, 30))
            rows.append(row)
    path = output/'original_sensenova_alpha.pptx'; prs.save(path)
    for offset in range(0, len(rows), 4):
        group = rows[offset:offset+4]; sheet = Image.new('RGB', (1080, 370*len(group)), 'white')
        for i, row in enumerate(group): sheet.paste(row, (0, 370*i))
        sheet.save(output/f'comparison_{offset//4+1:02d}.jpg', quality=92)
    save(output/'page_manifest.json', pages); save(output/'asset_manifest.json', records); save(output/'case_manifest.json', cases)
    summary = {'cases': len(manifest['cases']), 'slides': len(pages), 'inventory': len(records),
               'generated': sum(bool(r['rgba']) for r in records), 'status_counts': dict(Counter(r['status'] for r in records)),
               'failed_or_pending': [r for r in records if not r['rgba']],
               'inventory_unavailable_cases': [c['case'] for c in cases if not c['inventory_available']],
              'unresolved_coverage_cases': [c['case'] for c in cases if c['coverage_status']=='unresolved'],
               'unresolved_boundary_cases': [c['case'] for c in cases if c['boundary_review_status']=='unresolved'],
               'visual_fidelity_verified': False}
    save(output/'VALIDATION.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(build_report(args.batch, args.output), ensure_ascii=False))
