"""Allow the existing line reread to correct a fallible baseline count."""
from pathlib import Path
import ast

def install(backend):
    path=Path(backend)/'src/shape_integration/texts/multiline_retry.py'
    source=path.read_text()
    replacements=[
        ("if not isinstance(lines,list) or len(lines)!=row['visual_line_count']:",
         "if not isinstance(lines,list) or not 1<=len(lines)<=64:"),
        ("'expected_lines':row['visual_line_count']", "'previous_line_count_hint':row['visual_line_count']"),
        ("'If the supplied content or line count cannot be located, return lines=[]; do not guess.'",
         "'The previous line count may include a neighbor. Return the actual baselines of this occurrence, including one if it is single-line. Preserve the full supplied transcription; if it cannot be located, return lines=[].'"),
    ]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('Inspect existing multiline recovery before adapting')
        source=source.replace(before,after,1)
    ast.parse(source);path.write_text(source)
