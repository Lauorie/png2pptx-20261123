"""The arrow branch works on as many images at once as the shape branch is told to."""
from full_recovery.arrow_workers import widened


def test_image_pools_grow_to_the_requested_number_of_images() -> None:
    config = {'task_workers': 8, 'case_workers': 2, 'cpu_workers': 2, 'sam_workers': 2}
    assert widened(config, 48, images=6) == {'task_workers': 24, 'case_workers': 6, 'cpu_workers': 6,
                                             'sam_workers': 2}


def test_without_a_number_of_images_the_image_pools_are_left_alone() -> None:
    config = {'task_workers': 8, 'case_workers': 2, 'cpu_workers': 2}
    assert widened(config, 48)['case_workers'] == 2
    assert widened(config, 48, images=None)['cpu_workers'] == 2


def test_a_wider_image_pool_is_not_narrowed() -> None:
    assert widened({'case_workers': 8, 'cpu_workers': 4}, 48, images=6) == {'case_workers': 8, 'cpu_workers': 6}


def test_image_pools_a_stage_does_not_define_are_not_added() -> None:
    assert widened({'api_concurrency': 48}, 48, images=6) == {'api_concurrency': 48}
