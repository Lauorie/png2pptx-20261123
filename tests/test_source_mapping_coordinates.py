"""A mapping batch must describe the pixels actually sent to the model."""
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from full_recovery.source_ownership import detail_coordinates, select_mapping, summary


def test_detail_transform_keeps_all_evidence_in_one_grid_without_mutating_ledger():
    content = {
        'source_objects': [{'id': 'S1', 'bbox': [100, 50, 120, 70]}],
        'candidates': [{'id': 'C1', 'points': [[100, 50], [120, 70]]}],
        'image_owned_regions': [{'id': 'I1', 'bbox': [100, 50, 120, 70]}],
        'prior_source_selection': [{'id': 'S1', 'kept_ids': ['C0'],
                                    'native': [{'id': 'C0', 'points': [[100, 50], [120, 70]]}]}]}
    before = deepcopy(content)
    result = detail_coordinates(content, [90, 40, 130, 80], [120, 100])
    assert result['source_objects'][0]['bbox'] == [31, 25.75, 91, 75.75]
    assert result['image_owned_regions'][0]['bbox'] == result['source_objects'][0]['bbox']
    assert result['candidates'][0]['points'] == result['prior_source_selection'][0]['native'][0]['points']
    assert result['coordinate_frame']['size'] == [120, 100]
    assert result['prior_source_selection'][0]['kept_ids'] == ['C0']
    assert content == before


def test_batched_mapper_points_land_on_supplied_source_pixels(tmp_path):
    image = Image.new('RGB', (600, 200), 'white')
    ImageDraw.Draw(image).line((20, 70, 140, 70), fill='black', width=3)
    source = tmp_path/'source.png'; image.save(source)
    aliases = {}
    for index in range(66):
        ident = 'C%d' % index
        aliases[ident] = {'id': 'shape:%d' % index, 'layer': 'shape', 'refs': [],
            'length': 120., 'points': np.c_[np.linspace(20, 140, 60), np.full(60, 70.)],
            'style': {'stroke': '#000000', 'width_px': 1}}
    objects = [{'id': 'S1', 'kind': 'line', 'bbox': [20, 69, 140, 71],
                'visible_count': 1, 'description': 'horizontal source occurrence'}]
    prompt = {'source_objects': objects, 'rules': [], 'output': {'objects': [{}]},
              'candidates': [summary(row, ident, [0, 0, 600, 200], np.ones(2))
                             for ident, row in aliases.items()]}
    before = deepcopy(prompt)
    class Mapper:
        def call(self, folder, tag, system, payload, images, validator, budget):
            with Image.open(images[0][1]) as displayed:
                assert displayed.size == tuple(payload['coordinate_frame']['size'])
                x, y = np.rint(payload['candidates'][0]['points'][4]).astype(int)
                assert max(displayed.getpixel((x, y))) < 30
            ids = [row['id'] for row in payload['candidates']]
            prior = payload.get('prior_source_selection', [{'kept_ids': []}])[0]['kept_ids']
            return validator({'objects': [{'id': 'S1', 'candidate_ids': ids,
                'keep_ids': ids[:1] if not prior else [], 'status': 'resolved',
                'reason': 'one physical source occurrence'}], 'unassigned': []})
    mapped, calls = select_mapping(Mapper(), tmp_path/'mapping', prompt,
        [('CLEAN', source)], objects, aliases, {}, [0, 0, 600, 200], np.ones(2))
    assert calls == 2 and mapped['objects'][0]['keep_ids'] == ['C0']
    assert len(mapped['objects'][0]['candidate_ids']) == 66
    assert prompt == before
