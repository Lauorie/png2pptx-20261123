"""The text reader asks a vision model for a figure's text lines and turns its reply into lines."""
from typing import Any, Dict, List

import pytest
from PIL import Image

from full_recovery.text_reader import ReaderConfig, ReadingFailed, make_reader, parse_reply


def test_every_row_of_the_reply_is_a_line_of_text() -> None:
    assert parse_reply('Skill S\n\n  3 layers  \nlabel1|label2|"n"') == ['Skill S', '3 layers', 'label1|label2|"n"']


def test_a_fenced_reply_is_read() -> None:
    assert parse_reply('```text\nLoss\n```') == ['Loss']


def test_a_figure_without_text_reads_as_no_lines() -> None:
    assert parse_reply('NONE') == []


def test_an_empty_reply_fails() -> None:
    for reply in ('', '  \n', '```\n```'):
        with pytest.raises(ReadingFailed):
            parse_reply(reply)


def test_the_reader_sends_the_picture_to_the_configured_model_and_retries() -> None:
    sent: List[Dict[str, Any]] = []

    def post(url: str, key: str, body: Dict[str, Any]) -> Dict[str, Any]:
        sent.append({'url': url, 'key': key, 'body': body})
        content = '' if len(sent) == 1 else 'Loss\nAccuracy'
        return {'choices': [{'message': {'content': content}}]}

    config = ReaderConfig(base_url='https://gateway.example/v1', key='k', model='vendor/reader', attempts=3)
    lines = make_reader(config, post)(Image.new('RGB', (200, 100), 'white'))
    assert lines == ['Loss', 'Accuracy']
    assert len(sent) == 2 and sent[0]['url'] == 'https://gateway.example/v1/chat/completions'
    assert sent[0]['body']['model'] == 'vendor/reader' and sent[0]['key'] == 'k'
    assert sent[0]['body']['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


def test_the_reader_gives_up_after_the_configured_attempts() -> None:
    def post(url: str, key: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {'choices': [{'message': {'content': ''}}]}

    config = ReaderConfig(base_url='u', key='k', model='m', attempts=2)
    with pytest.raises(ReadingFailed):
        make_reader(config, post)(Image.new('RGB', (20, 10), 'white'))


def test_of_several_readings_the_one_the_others_agree_with_most_is_kept() -> None:
    from full_recovery.text_reader import consensus
    full, cut = ['Seed Workflows', 'General Agent', 'Memory Archive'], ['Seed Workflows']
    by_width = {100: full, 150: cut, 200: full[::-1]}
    seen: List[int] = []

    def read(image: Image.Image) -> List[str]:
        seen.append(image.width)
        return by_width[image.width]

    picture = Image.new('RGB', (100, 50), 'white')
    assert sorted(consensus(read, (1., 1.5, 2.))(picture)) == sorted(full)
    assert sorted(seen) == [100, 150, 200]
    by_width[100] = cut
    by_width[150] = full
    assert sorted(consensus(read, (1., 1.5, 2.))(picture)) == sorted(full)


def test_a_reading_that_fails_is_left_out_of_the_consensus() -> None:
    from full_recovery.text_reader import consensus

    def read(image: Image.Image) -> List[str]:
        if image.width == 150:
            raise ReadingFailed('no')
        return ['Loss']

    assert consensus(read, (1., 1.5, 2.))(Image.new('RGB', (100, 50), 'white')) == ['Loss']


def test_consensus_fails_when_every_reading_fails() -> None:
    from full_recovery.text_reader import consensus

    def read(image: Image.Image) -> List[str]:
        raise ReadingFailed('no')

    with pytest.raises(ReadingFailed):
        consensus(read, (1., 2.))(Image.new('RGB', (100, 50), 'white'))
