"""Conservative repeated-label diagnostics, never an automatic style gate."""
from statistics import median


def audit(rows):
    warnings=[]
    eligible=[r for r in rows if r.get('kind')!='formula' and r.get('font_size_px',0)>0
              and r.get('visual_line_count',1)==1 and len(r.get('text',''))<=4
              and (r.get('text','').isdecimal() or r.get('text','').isalpha())]
    for row in eligible:
        x0,y0,x1,y1=row['bbox_2d'];height=y1-y0+1;width=x1-x0+1
        peers=[]
        for other in eligible:
            if other['id']==row['id'] or other.get('font')!=row.get('font'):
                continue
            if abs(other.get('rotation_deg',0)-row.get('rotation_deg',0))>1:
                continue
            if row['text'].isdecimal()!=other['text'].isdecimal():
                continue
            # Alphabetic labels must be repeated glyphs, not arbitrary prose.
            if not row['text'].isdecimal() and row['text']!=other['text']:
                continue
            a,b,c,d=other['bbox_2d'];oh=d-b+1
            aligned=(abs((y0+y1-b-d)/2)<.6*max(height,oh) and abs((x0+x1-a-c)/2)<30*max(height,oh))
            aligned|=(abs((x0+x1-a-c)/2)<max(width,c-a+1) and abs((y0+y1-b-d)/2)<30*max(height,oh))
            if aligned:peers.append(other)
        if len(peers)<3:continue
        expected=median(r['font_size_px'] for r in peers)
        # At least three agreeing peers and a large outlier. Report only;
        # nested diagram levels may legitimately use different typography.
        agreeing=[r['id'] for r in peers if .85<=r['font_size_px']/expected<=1.15]
        ratio=row['font_size_px']/expected
        if len(agreeing)>=3 and (ratio>1.35 or ratio<1/1.35):
            warnings.append({'stage':'typography_consistency','id':row['id'],
                'reason':'Possible repeated-label font-size outlier; style retained for review',
                'font_size_px':row['font_size_px'],'peer_median_px':expected,'peer_ids':agreeing})
    return warnings
