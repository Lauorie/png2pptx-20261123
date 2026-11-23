"""Preserve run/break insertion order in the bundled python-pptx XML schema.

Its add_r inserts runs before every existing br, making a stacked label one
horizontal word followed by empty lines. Apply only inside the text worker;
do not modify the shared environment or the evaluator's libraries.
"""
from functools import wraps


def install():
    from pptx.oxml.text import CT_TextParagraph
    for name in ('add_r','add_br'):
        original=getattr(CT_TextParagraph,name)
        if getattr(original,'_native_append_order',False):continue
        def ordered(method):
            @wraps(method)
            def append(self,*args,**kwargs):
                node=method(self,*args,**kwargs)
                end=self.endParaRPr
                if end is None:self.append(node)
                else:end.addprevious(node)
                return node
            append._native_append_order=True
            return append
        setattr(CT_TextParagraph,name,ordered(original))
