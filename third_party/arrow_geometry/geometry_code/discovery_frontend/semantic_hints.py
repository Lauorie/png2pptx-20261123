"""Do not carry unframed numerical positions inside semantic hint strings."""
import copy
import re

HINT_KEYS={'locator','locator_hint','start_hint','end_hint','route_hint','identity_hint'}
NUMBER=r'-?\d+(?:\.\d+)?'
AXIS=re.compile(r'(?<![A-Za-z0-9])(?:x|y|横坐标|纵坐标)\s*[≈=~:：]\s*'+NUMBER+r'(?:\s*[-–—~至到]\s*'+NUMBER+r')?',re.I)
POINT=re.compile(r'(?:约|≈|~)\s*[（(]\s*'+NUMBER+r'\s*[,，]\s*'+NUMBER+r'\s*[)）]')

def semantic_text(value):
    if not isinstance(value,str):return value
    value=AXIS.sub('',POINT.sub('',value))
    value=re.sub(r'[（(]\s*[,，;；、\s]*[)）]','',value)
    value=re.sub(r'\s+',' ',value).strip(' ,，;；')
    return value or 'unknown'

def clean_prompt(prompt):
    """Return a new prompt and a separate raw-to-clean audit; numeric arrays stay exact."""
    changes=[]
    def visit(value,path):
        if isinstance(value,dict):
            result={}
            for k,v in value.items():
                if k in HINT_KEYS and isinstance(v,str):
                    result[k]=semantic_text(v)
                    if result[k]!=v:changes.append({'path':path+[k],'before':v,'after':result[k],'reason':'unframed_coordinate_text_withheld'})
                else:result[k]=visit(v,path+[k])
            return result
        if isinstance(value,list):return [visit(v,path+[i]) for i,v in enumerate(value)]
        return copy.deepcopy(value)
    return visit(prompt,[]),changes
