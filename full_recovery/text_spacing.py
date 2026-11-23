"""Normalize OCR horizontal separators before native font measurement.

A tab has renderer-defined stops, not a source glyph width. Ordinary single
baseline labels use spaces consistently in text, runs, measurement and export.
Explicit multiline/stacked/formula layout keeps its original structure.
"""
from copy import deepcopy


def normalize(row):
    if (row.get('kind','text')!='text' or row.get('layout_mode','single_line')!='single_line'
            or row.get('visual_line_count',1)!=1 or '\t' not in row.get('text','')):
        return row
    out=deepcopy(row);out['text']=out['text'].replace('\t',' ')
    for run in out.get('runs',[]):run['text']=run.get('text','').replace('\t',' ')
    out['horizontal_separator_normalization']='single_baseline_tab_to_space_before_measurement'
    return out


def install():
    from shape_integration.texts import layout
    if getattr(layout.fit_all,'_source_horizontal_spacing',False):return
    original=layout.fit_all
    def fit_all(rows,*args,**kwargs):
        return original([normalize(row) for row in rows],*args,**kwargs)
    fit_all._source_horizontal_spacing=True
    layout.fit_all=fit_all
