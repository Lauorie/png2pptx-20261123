from copy import deepcopy

from full_recovery.text_inventory import install
from full_recovery.text_diagnostic_repair import ask_inventory, formula_structure_conflicts, repeated_style_diagnostics
from shape_integration.texts import options, inventory_audit, model


def occurrence(text, box, replaces):
    return {'kind':'text', 'runs':[{'text':text,'script':'normal','color':'#000000'}],
            'bbox_2d':box, 'font':'Arial','color':'#000000','legibility':'clear',
            'replaces':replaces,'rotation_deg':0,'visual_line_count':1}


def test_malformed_neighbor_retains_valid_edit_and_only_rereads_rejected(tmp_path, monkeypatch):
    install()
    opt=options({})
    valid=occurrence('correct',[10,10,80,30],['old'])
    invalid=occurrence('',[90,10,150,30],[])
    corrected=occurrence('missing',[90,10,150,30],[])
    validate=lambda raw:inventory_audit.validate_reply(raw,[200,100],opt,['old'],[])
    raw={'texts':[valid,invalid],'regions':[],'uncertain':[]}
    before=deepcopy(raw)
    checked=validate(raw)
    assert len(checked['texts'])==1 and len(checked['rejected_texts'])==1
    assert raw==before
    prompts=[]
    def ask(runtime, folder, prompt, images, validator, opt):
        prompts.append(prompt)
        return validator(raw if len(prompts)==1 else {'texts':[corrected]})
    monkeypatch.setattr(model,'ask',ask)
    reply=ask_inventory(object(),tmp_path,{'rules':[]},[],validate,opt)
    assert len(prompts)==2
    assert len(prompts[1]['rejected_observations'])==1
    assert [r['text'] for r in reply['texts']]==['correct','missing']
    assert not reply['rejected_texts']
    assert reply['diagnostic_recovery']['state']=='recovered'


def test_failed_bounded_reread_does_not_lose_valid_inventory_edits(tmp_path, monkeypatch):
    install();opt=options({})
    raw={'texts':[occurrence('kept',[10,10,80,30],['old']),
                  occurrence('',[90,10,150,30],[])]}
    validate=lambda value:inventory_audit.validate_reply(value,[200,100],opt,['old'],[])
    calls=[]
    def ask(runtime, folder, prompt, images, validator, opt):
        calls.append(folder)
        if len(calls)>1:raise RuntimeError('Transport unavailable')
        return validator(raw)
    monkeypatch.setattr(model,'ask',ask)
    reply=ask_inventory(object(),tmp_path,{'rules':[]},[],validate,opt)
    assert [r['text'] for r in reply['texts']]==['kept'] and len(calls)==2
    assert reply['diagnostic_recovery']['state']=='needs_review'


def test_formula_source_disagreement_nominates_existing_reread_without_rewrite():
    rows=[{'id':'a','kind':'formula','latex':'q_{7}'},
          {'id':'b','kind':'formula','latex':'q_{8}'},
          {'id':'plain','kind':'text','text':'unchanged'}]
    updates={'a':[{**rows[0],'latex':'q^{7}'}],
             'b':[{**rows[1],'latex':'{q}_{8}'}]}
    before=deepcopy((rows,updates))
    events=formula_structure_conflicts(rows,updates)
    assert [e['id'] for e in events]==['a']
    assert (rows,updates)==before


def test_repeated_weight_diagnostics_abstain_for_different_levels_and_ties():
    rows=[{'id':str(i),'kind':'text','text':'repeated','bbox_2d':[i*60,0,i*60+50,12],
           'bold':i==0,'rotation_deg':0}for i in range(5)]
    before=deepcopy(rows)
    assert [e['id'] for e in repeated_style_diagnostics(rows)]==['0']
    assert rows==before
    rows[0]['bbox_2d'][3]=24
    assert not repeated_style_diagnostics(rows)
    rows=before
    rows[1]['bold']=True;rows[2]['bold']=True
    assert not repeated_style_diagnostics(rows)


def test_geometry_boundary_is_shared_by_global_and_local_inventory():
    from shape_integration.texts.staged_formula import inventory_prompt
    install()
    opt=options({})
    for local in (False,True):
        prompt=inventory_prompt([200,100],opt,local=local,known=['label'] if local else [])
        rules=[r for r in prompt['rules']if r.startswith('GEOMETRY BOUNDARY:')]
        assert len(rules)==1
        assert 'node bodies and port markers' in rules[0]
        assert 'CLEAN' in rules[0]


def test_composite_formula_character_disagreement_uses_existing_recovery():
    rows=[{'id':'formula','kind':'formula','latex':r'\textcolor{#777777}{W^{\mathrm{assoc}}(t)}'}]
    updated={'formula':[{**rows[0],'latex':r'M^{\mathrm{assoc}}(t)'}]}
    assert [r['id']for r in formula_structure_conflicts(rows,updated)]==['formula']
    same={'formula':[{**rows[0],'latex':r'W^{\mathrm{assoc}}(t)'}]}
    assert formula_structure_conflicts(rows,same)==[]


def test_formula_word_packing_does_not_schedule_redundant_reread():
    rows=[{'id':'formula','kind':'formula','latex':r'z^{\mathrm{value}}(t-d_{\mathrm{feedback}})'}]
    updates={'formula':[{**rows[0],'latex':r'z^{\text{value}}(t-d_{\text{feedback}})'}]}
    assert formula_structure_conflicts(rows,updates)==[]
