"""Keep the reference text recovery in the original occurrence's neighborhood."""
import hashlib
from pathlib import Path


def install(backend):
    path = Path(backend)/'src/shape_integration/texts/two_stage_crop.py'
    source = path.read_text()
    before = hashlib.sha256(source.encode()).hexdigest()
    replacements = [
        ("POLICY = 'two_stage_crop_visual_recall_wide_recovery_no_cpu_gate_v5'",
         "POLICY = 'reference_text_expanded_local_recovery.1'"),
        ("        # A target outside its first crop cannot be recovered by another prompt\n"
         "        # over the same pixels. Full-source recovery supplies the missing context.\n"
         "        f=frame(source,folder/'input',{**opt,'crop_min_padding_px':max(64,short*2.5),'crop_padding':0,'crop_scale':100},\n"
         "                None if wide else union(group))",
         "        # Expand the same physical neighborhood at local glyph resolution.\n"
         "        padding=max(64,short*2.5)*(2 if wide else 1)\n"
         "        f=frame(source,folder/'input',{**opt,'crop_min_padding_px':padding,'crop_padding':0,'crop_scale':100},\n"
         "                union(group))"),
        ('FULL-FIGURE RECOVERY: previous rough boxes may point at blank space or the wrong diagram node. Locate the complete supplied occurrence from CLEAN, using its text and surrounding structure. Do not remain inside the rough box. Never substitute a different occurrence of a repeated label. If identity is unresolved, return ambiguous; do not guess.',
         'EXPANDED LOCAL RECOVERY: locate the same physical occurrence using its original position and neighboring structure. Previous spelling and boxes are fallible. Search beyond the rough box, but never substitute a matching label from another row or node. If this view cannot resolve the original occurrence, return ambiguous.'),
        ("'scope':'full_figure_recovery' if wide else 'local_crop'",
         "'scope':'expanded_local_recovery' if wide else 'local_crop'"),
        ("            parts=updates.get(row['id'])\n"
         "            # Parent IDs are stable; a split parent is reread as a complete unit.\n"
         "            recovery_rows.append(parts[0] if parts and len(parts)==1 else row)",
         "            # Do not let a mistaken first update move the recovery anchor.\n"
         "            recovery_rows.append(deepcopy(row))")]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Reference text scope adaptation no longer matches source')
        source = source.replace(old, new)
    path.write_text(source)
    return {'protocol':'reference_text_expanded_local_recovery.1',
            'file':str(path.relative_to(backend)), 'before_sha256':before,
            'after_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'change':'Only the existing recovery crop and its stable spatial anchor',
            'extra_model_stages':0, 'sample_specific_rules':False}
