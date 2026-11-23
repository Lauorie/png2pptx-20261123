from full_recovery.quality_choice import enclosing_context_keys


def closed(points):
    return {'kind': 'path', 'commands': [
        ['M', *points[0]], *[['L', x, y] for x, y in points[1:]], ['Z']]}


def test_native_enclosure_is_context_without_promoting_partial_faces_or_candidates():
    parts = {'body': closed([(0, 0), (100, 0), (100, 80), (0, 80)]),
             'partial': closed([(10, 10), (50, 10), (50, 40), (10, 40)]),
             'candidate': closed([(0, 0), (100, 0), (100, 80), (0, 80)]),
             'transparent': closed([(0, 0), (100, 0), (100, 80), (0, 80)])}
    styles = {key: {'fill': '#185F77'} for key in parts}
    styles['transparent']['fill'] = None
    rows = [{'keys': ['candidate'], 'points': [(20, 25), (70, 25)]}]
    assert enclosing_context_keys(rows, {'parts': parts}, styles) == {'body'}


def test_bounding_box_alone_does_not_establish_a_painted_enclosure():
    parts = {'triangle': closed([(0, 0), (100, 0), (0, 100)])}
    rows = [{'keys': [], 'points': [(60, 70), (80, 70)]}]
    assert enclosing_context_keys(rows, {'parts': parts},
        {'triangle': {'fill': '#185F77'}}) == set()
