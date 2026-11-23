"""Conservative equivalence of single-base script display and structured runs."""
import unicodedata


def normalize(value):
    return ''.join(unicodedata.normalize('NFKC',value).replace('\u2212','-').split())


def unicode_script(char):
    parts=unicodedata.decomposition(char).split()
    if parts and parts[0] in ('<super>','<sub>'):
        return ('super' if parts[0]=='<super>' else 'sub', ''.join(chr(int(p,16)) for p in parts[1:]))
    return None


def display_structure(text):
    """Parse only a base followed by attached scripts, not general LaTeX."""
    base=[];scripts={};index=0;started=False;explicit=set()
    while index<len(text):
        char=text[index]
        if char.isspace():index+=1;continue
        special=unicode_script(char)
        if special:
            if not base:return None
            kind,value=special;scripts[kind]=scripts.get(kind,'')+value
            started=True;index+=1;continue
        if char in '_^':
            if not base:return None
            kind='sub' if char=='_' else 'super'
            if kind in explicit or kind in scripts:return None
            explicit.add(kind);index+=1
            while index<len(text) and text[index].isspace():index+=1
            if index>=len(text):return None
            if text[index]=='{':
                end=text.find('}',index+1)
                if end<0 or any(c in text[index+1:end] for c in '{}\\'):return None
                value=text[index+1:end];index=end+1
            else:
                value=text[index];index+=1
                if value in '{}_^\\':return None
            if not normalize(value):return None
            scripts[kind]=value;started=True;continue
        if started or char in '{}\\':return None
        base.append(char);index+=1
    if not base or not scripts:return None
    return normalize(''.join(base)),{k:normalize(v) for k,v in scripts.items()}


def runs_structure(runs):
    base=[];scripts={};started=False
    for run in runs:
        kind=run.get('script');value=run.get('text')
        if not isinstance(value,str):return None
        if not normalize(value):continue
        if kind=='normal':
            if started:return None
            base.append(value)
        elif kind in ('sub','super'):
            if not base:return None
            started=True;scripts[kind]=scripts.get(kind,'')+value
        else:return None
    if not base or not scripts:return None
    return normalize(''.join(base)),{k:normalize(v) for k,v in scripts.items()}


def equivalent(text,runs):
    observed=display_structure(text)
    return observed is not None and observed==runs_structure(runs)
