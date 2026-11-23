from full_recovery.text_product import exact_audit_duplicates


def row(ident, origin, text='omega', box=(10,20,30,80), **extra):
    return dict(id=ident, origin=origin, kind='text', text=text,
                bbox_2d=list(box), rotation_deg=0, **extra)


def test_audit_alias_drops_after_a_local_owner_has_corrected_typography():
    owner=row('localized.G02','staged_global',font_size_px=18)
    owner['rotation_deg']=-90
    recall=row('recalled','inventory_visual_audit',font_size_px=3)
    drops,audit=exact_audit_duplicates([recall,owner])
    assert drops==['recalled'] and audit[0]['owner_text_id']=='localized.G02'
    assert owner['rotation_deg']==-90 and owner['font_size_px']==18


def test_same_words_at_separate_source_locations_and_partial_words_survive():
    owner=row('localized','staged_global',text='Detection')
    rows=[owner,row('elsewhere','inventory_visual_audit',text='Detection',box=(40,20,60,80)),
          row('partial','inventory_visual_audit',text='De'),
          row('full','inventory_visual_audit',text='Detection',box=(9,20,30,80))]
    assert exact_audit_duplicates(rows)==([],[])


def test_two_localized_occurrences_are_not_deleted_by_audit_provenance_rule():
    assert exact_audit_duplicates([row('first','staged_global'),row('second','local_recovery')])==([],[])
