"""Latest valid LaTeX owns formula content/colors; retain fallback provenance."""
from copy import deepcopy

from .omml import latex_to_omml, normalize_latex_input, OMML_NS


def adopt(row, item):
    result=deepcopy(row)
    latex=normalize_latex_input(item['latex'])
    tree=latex_to_omml(latex,color=row['color'])
    runs=[]
    for node in tree.xpath('.//*[local-name()="r" and namespace-uri()="'+OMML_NS+'"]'):
        text=''.join(node.xpath('./*[local-name()="t"]/text()'))
        if not text:continue
        colors=node.xpath('.//*[local-name()="srgbClr"]/@val')
        color='#'+colors[0].upper() if colors else row['color']
        script='normal'
        for ancestor in node.iterancestors():
            if ancestor.tag=='{'+OMML_NS+'}sub':script='sub';break
            if ancestor.tag=='{'+OMML_NS+'}sup':script='super';break
        if runs and runs[-1]['script']==script and runs[-1]['color']==color:runs[-1]['text']+=text
        else:runs.append({'text':text,'script':script,'color':color})
    if not runs:raise ValueError('Formula has no visible leaf content')
    # This is display metadata, not an alternate parser for mathematical layout.
    result.update(latex=latex,text=''.join(r['text'] for r in runs),runs=runs,
        formula_status='resolved',formula_evidence=deepcopy(item),line_layout_status='native_formula',
        formula_authority='latest_valid_latex',visual_line_count=1)
    if result['text']!=row['text'] or runs!=row['runs'] or latex!=row.get('latex',''):
        result['formula_revision']={'policy':'latest_valid_reply_wins',
            'before':{k:deepcopy(row.get(k)) for k in ('text','runs','latex')},
            'after':{k:deepcopy(result.get(k)) for k in ('text','runs','latex')}}
    return result


def keep_previous(row, reason):
    """Keep valid prior syntax, else only known runs as explicitly partial math.

    The fallback preserves known characters, colors and scripts in OMML. It
    does NOT infer missing fraction/radical hierarchy and is never 'resolved'.
    """
    result=deepcopy(row)
    if row.get('latex'):
        try:
            latex_to_omml(row['latex'],color=row['color'])
            result['formula_fallback']={'policy':'previous_valid_latex','reason':reason}
            return result
        except (ValueError,RuntimeError):pass
    def literal(value):
        # Backslashes are literal math glyphs, never executable TeX from text.
        escape=lambda s:''.join('\\'+c if c in '{}%_&#$' else c for c in s)
        return r'\backslash '.join(r'\text{'+escape(s)+'}' for s in value.split('\\'))
    atoms=[]
    for run in row['runs']:
        value=r'\textcolor{'+run.get('color',row['color'])+'}{'+literal(run['text'])+'}'
        script=run.get('script','normal')
        if script=='normal':atoms.append({'base':value,'sub':'','super':''})
        else:
            if not atoms:atoms.append({'base':'{}','sub':'','super':''})
            atoms[-1][script]+=value
    latex=''.join(a['base']+('_{'+a['sub']+'}' if a['sub'] else '')+
                  ('^{'+a['super']+'}' if a['super'] else '') for a in atoms)
    latex_to_omml(latex,color=row['color'])
    result.update(latex=latex,formula_status='fallback_runs',line_layout_status='native_formula',
        formula_fallback={'policy':'known_runs_only_no_inferred_hierarchy','reason':reason,
                          'requires_review':True})
    return result
