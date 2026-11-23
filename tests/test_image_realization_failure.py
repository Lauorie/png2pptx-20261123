from PIL import Image


def test_empty_owned_scope_keeps_native_owner(tmp_path):
    from full_recovery.image_realization import realize

    source = tmp_path / 'source.png'
    Image.new('RGB', (10, 10), 'white').save(source)
    output = tmp_path / 'edited.png'

    class Editor:
        def edit(self, crop, request, path, role=None):
            crop.convert('RGBA').save(path)
            return {
                'source_offset_xy': [0, 0],
                'source_extent_wh': list(crop.size),
                'requests': 1,
            }

    asset = {
        'id': 'asset', 'path': str(output), 'pending_edit': True,
        'edit_context_half_open': [0, 0, 10, 10], 'edit_request': {},
        'role': 'icon', 'replace_parts': ['p'],
        'edit': {
            'owned_box': [20, 20, 30, 30],
            'preserved_native_frames': [],
            'separate_foreground_parts': [],
        },
    }
    texture = {'assets': [asset], 'drop_shape_keys': ['p']}
    scene = {
        'source_size': [10, 10],
        'parts': {'p': {'bbox_2d': [1, 1, 2, 2], 'commands': []}},
        'objects': {},
    }

    result = realize(texture, scene, {}, source, {}, Editor())

    assert result['assets'] == []
    assert result['drop_shape_keys'] == []
    assert result['realization_failures'][0]['fallback'] == 'retain_native_owner'
