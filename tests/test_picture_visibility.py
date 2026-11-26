"""A picture placed as a backdrop must not end up hidden under an opaque shape."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor'), str(ROOT / 'experiment/shape_sam_guided')]

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu

from full_recovery.image_assets import commit
from full_recovery.texture_layer import above_covering, opaque_boxes

SIZE = (400, 200)
UNIT = 9525  # EMU per source pixel in these decks


def test_the_picture_goes_above_the_last_sibling_that_covers_it() -> None:
    covers = [[], [(0, 0, 100, 100)], [(50, 50, 60, 60)], [(5, 5, 40, 40)], []]
    assert above_covering((10, 10, 20, 20), covers, default=0) == 4


def test_without_a_covering_sibling_the_default_layer_is_kept() -> None:
    assert above_covering((10, 10, 20, 20), [[], [(50, 50, 60, 60)]], default=1) == 1


def test_a_sibling_that_hides_only_part_of_the_picture_does_not_count() -> None:
    assert above_covering((10, 10, 20, 20), [[(0, 0, 15, 100)]], default=0) == 0


def _deck(tmp_path: Path, shape: MSO_SHAPE = MSO_SHAPE.RECTANGLE, filled: bool = True, grouped: bool = False) -> Path:
    deck = Presentation()
    deck.slide_width, deck.slide_height = Emu(SIZE[0] * UNIT), Emu(SIZE[1] * UNIT)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    holder = slide.shapes.add_group_shape().shapes if grouped else slide.shapes
    panel = holder.add_shape(shape, Emu(20 * UNIT), Emu(20 * UNIT), Emu(300 * UNIT), Emu(160 * UNIT))
    panel.name = 'G001/panel'
    if filled:
        panel.fill.solid()
        panel.fill.fore_color.rgb = RGBColor(0xE8, 0xE8, 0xE8)
    else:
        panel.fill.background()
    label = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(330 * UNIT), Emu(20 * UNIT), Emu(50 * UNIT), Emu(30 * UNIT))
    label.name = 'G002/label'
    path = tmp_path / 'shapes.pptx'
    deck.save(path)
    return path


def _children(path: Path) -> list:
    tree = Presentation(str(path)).slides[0].shapes._spTree
    return [opaque_boxes(node) for node in tree]


def test_a_filled_rectangle_is_opaque_over_its_box(tmp_path: Path) -> None:
    boxes = [b for child in _children(_deck(tmp_path)) for b in child]
    assert (20 * UNIT, 20 * UNIT, 320 * UNIT, 180 * UNIT) in boxes


def test_an_unfilled_rectangle_hides_nothing(tmp_path: Path) -> None:
    boxes = [b for child in _children(_deck(tmp_path, filled=False)) for b in child]
    assert (20 * UNIT, 20 * UNIT, 320 * UNIT, 180 * UNIT) not in boxes


def test_an_ellipse_is_not_assumed_to_cover_its_box(tmp_path: Path) -> None:
    boxes = [b for child in _children(_deck(tmp_path, shape=MSO_SHAPE.OVAL)) for b in child]
    assert (20 * UNIT, 20 * UNIT, 320 * UNIT, 180 * UNIT) not in boxes


def test_a_filled_rectangle_inside_a_group_is_found(tmp_path: Path) -> None:
    boxes = [b for child in _children(_deck(tmp_path, grouped=True)) for b in child]
    assert (20 * UNIT, 20 * UNIT, 320 * UNIT, 180 * UNIT) in boxes


def _commit(tmp_path: Path, deck: Path) -> list:
    Image.new('RGB', (60, 60), '#336699').save(tmp_path / 'photo.png')
    asset = {'id': 'U1', 'owner': None, 'role': 'background', 'path': str(tmp_path / 'photo.png'),
             'sha256': hashlib.sha256((tmp_path / 'photo.png').read_bytes()).hexdigest(),
             'placement_source_half_open': [60, 60, 120, 120]}
    commit(deck, tmp_path / 'out.pptx', [asset], SIZE)
    return [shape.name for shape in Presentation(str(tmp_path / 'out.pptx')).slides[0].shapes]


def test_a_backdrop_picture_is_placed_above_the_panel_that_would_hide_it(tmp_path: Path) -> None:
    names = _commit(tmp_path, _deck(tmp_path))
    assert names.index('IMG U1') > names.index('G001/panel')
    assert names.index('IMG U1') < names.index('G002/label')


def test_a_backdrop_picture_with_nothing_over_it_stays_at_the_back(tmp_path: Path) -> None:
    names = _commit(tmp_path, _deck(tmp_path, filled=False))
    assert names.index('IMG U1') == 0
