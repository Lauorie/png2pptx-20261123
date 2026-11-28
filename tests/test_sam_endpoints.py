"""Several SAM instances can be named at once; they must all serve the same model."""
import pytest

from full_recovery.service_endpoints import sam_bases, same_sam_model


def test_one_address_is_a_list_of_one() -> None:
    assert sam_bases('http://127.0.0.1:18103/') == ['http://127.0.0.1:18103']


def test_addresses_are_split_on_commas_and_normalised() -> None:
    assert sam_bases(' http://a:18103/v1/segment , http://a:18104/v1,http://a:18105 ') == [
        'http://a:18103', 'http://a:18104', 'http://a:18105']


def test_a_repeated_address_counts_once_and_order_is_kept() -> None:
    assert sam_bases('http://a:2,http://a:1,http://a:2/') == ['http://a:2', 'http://a:1']


@pytest.mark.parametrize('value', ['', ' , ', ','])
def test_no_address_is_refused(value: str) -> None:
    with pytest.raises(ValueError):
        sam_bases(value)


def test_instances_of_one_model_are_accepted_and_the_first_is_returned() -> None:
    first = {'url': 'http://a:1', 'epoch': 7, 'sha256': 'x' * 64, 'checkpoint_path': '/m'}
    second = {'url': 'http://a:2', 'epoch': 7, 'sha256': 'x' * 64, 'checkpoint_path': '/other/m'}
    assert same_sam_model([first, second]) is first


def test_an_instance_serving_other_weights_is_refused() -> None:
    first = {'url': 'http://a:1', 'epoch': 7, 'sha256': 'x' * 64}
    other = {'url': 'http://a:2', 'epoch': 7, 'sha256': 'y' * 64}
    with pytest.raises(ValueError, match='http://a:2'):
        same_sam_model([first, other])


def test_an_instance_at_another_epoch_is_refused() -> None:
    with pytest.raises(ValueError):
        same_sam_model([{'url': 'u1', 'epoch': 7, 'sha256': 'x'}, {'url': 'u2', 'epoch': 8, 'sha256': 'x'}])
