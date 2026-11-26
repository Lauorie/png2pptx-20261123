"""A texture with no native part to replace is layered behind what it contains."""
from full_recovery.texture_layer import behind_contained

SURFACE = (100, 100, 900, 700)


def test_the_texture_goes_just_below_the_first_object_it_contains():
    siblings = [None, None, (0, 0, 1000, 800), (200, 200, 260, 280), (950, 50, 990, 90), (400, 300, 470, 380)]
    assert behind_contained(SURFACE, siblings, default=len(siblings)) == 3


def test_a_backdrop_larger_than_the_texture_stays_below_it():
    siblings = [None, None, (0, 0, 1000, 800), (50, 50, 950, 750)]
    assert behind_contained(SURFACE, siblings, default=4) == 4


def test_without_contained_objects_the_default_layer_is_kept():
    siblings = [None, None, (950, 50, 990, 90)]
    assert behind_contained(SURFACE, siblings, default=3) == 3


def test_an_object_slightly_over_the_edge_still_counts_as_contained():
    siblings = [None, None, (95, 300, 160, 380)]
    assert behind_contained(SURFACE, siblings, default=3) == 2


def test_an_object_mostly_outside_does_not_count():
    siblings = [None, None, (20, 300, 140, 380)]
    assert behind_contained(SURFACE, siblings, default=3) == 3


def test_an_empty_box_is_ignored():
    siblings = [None, None, (300, 300, 300, 300)]
    assert behind_contained(SURFACE, siblings, default=3) == 3
