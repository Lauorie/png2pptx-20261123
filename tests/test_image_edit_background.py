"""The edit instruction pins a white canvas even when the target asks otherwise."""
from typing import Any, Dict

import pytest
import requests
from PIL import Image

from full_recovery.image_edit import Editor


class _Captured(Exception):
    """Raised by the fake session once the outgoing request is recorded."""


def _instruction(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, target: str) -> str:
    sent: Dict[str, Any] = {}

    def post(self: requests.Session, url: str, json: Dict[str, Any], timeout: Any) -> None:
        sent.update(json)
        raise _Captured()

    monkeypatch.setattr(requests.Session, 'post', post)
    editor = Editor({'base_url': 'http://127.0.0.1:9/v1'})
    prompt = {'target': target, 'remove_overlays': []}
    with pytest.raises(_Captured):
        editor.edit(Image.new('RGB', (40, 30), 'white'), prompt, tmp_path / 'out.png', role='texture')
    return sent['prompt']


def test_a_target_asking_for_transparency_is_overridden(monkeypatch, tmp_path):
    text = _instruction(monkeypatch, tmp_path, '保留曲面，去除白底并保留透明背景。')
    assert '忽略目标描述中关于透明背景的要求' in text
    assert '纯白' in text


def test_the_override_follows_the_target_description(monkeypatch, tmp_path):
    text = _instruction(monkeypatch, tmp_path, '橙蓝渐变曲面')
    assert text.index('橙蓝渐变曲面') < text.index('忽略目标描述中关于透明背景的要求')
