"""Offline regression: all independently copied model clients use bounded coordinates."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACKAGES = [
    ROOT / "transport",
    ROOT / "checkpoint/upstream/localization/code/transport",
    ROOT / "checkpoint/upstream/sam_supplement/code/transport",
    ROOT / "discovery_frontend/transport",
]


@pytest.fixture(params=range(len(PACKAGES)), ids=["geometry_style", "localization", "sam", "legacy_frontend"])
def package(request):
    path = PACKAGES[request.param]
    name = "_precision_transport_" + str(request.param)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                      submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


def test_coordinate_values_only_and_no_mutation(package):
    normalize = importlib.import_module(package + ".coordinate_precision").normalize_prompt
    prompt = {
        "existing": [{"bbox": [161.07608695652178, 30.597302504816938, 411.1847826086956, 145.2177263969171]}],
        "target_bbox": [0.0001, -0.0001, 1536.0, 540.0404],
        "crop_in_context": [1.12345, 2.23456, 6.12345, 8.34567],
        "current": {"nodes": [["n1", 12.34567, 34.56789]],
                    "points": [{"id": "p1", "xy": [1.234567, 2.345678]}]},
        "targets": [{"start": [12.34567, 34.56789], "end": [45.67891, 67.89123],
                     "via": [[15.12345, 40.45678]]}],
        "confidence": 0.987654321,
        "scale": 0.9755578741,
        "current_styles": [{"width_px": 2.3810754442817754, "start": {"type": "oval"}}],
        "label": "node 0.123456789",
    }
    before = copy.deepcopy(prompt)
    result = normalize(prompt)
    assert prompt == before
    assert result["existing"][0]["bbox"] == [161.1, 30.6, 411.2, 145.2]
    assert result["current"]["nodes"] == [["n1", 12.3, 34.6]]
    assert result["current"]["points"][0]["xy"] == [1.2, 2.3]
    assert result["targets"][0]["end"] == [45.7, 67.9]
    for key in ("confidence", "scale", "current_styles", "label"):
        assert result[key] == before[key]
    assert normalize(result) == result
    result["current"]["nodes"][0][1] = 999
    assert prompt == before


def test_text_prompts_and_verbatim_repairs(package):
    normalize = importlib.import_module(package + ".coordinate_precision").normalize_prompt
    original = '{"regions":[{"bbox":[1.234567,2.345678,30.456789,40.567891]}]}'
    candidates = json.dumps([{"id": "C1", "bbox": [1.234567, 2.345678, 30.456789, 40.567891]}])
    text = ("LOCAL 对应的 GLOBAL 观察范围：[0.0, 20.234567, 1080, 403.456789]。\n"
            "本批候选：\n" + candidates + "\n阈值 0.123456789 不变。")
    out = normalize(text)
    assert "[0.0, 20.2, 1080, 403.5]" in out
    assert '"bbox": [1.2, 2.3, 30.5, 40.6]' in out
    assert "阈值 0.123456789 不变。" in out
    repair = {"original_reply": original, "previous_output": original,
              "format_repair": {"previous_output": original},
              "schema": {"bbox": [1.234567, 2.345678, 30.456789, 40.567891]}}
    out = normalize(repair)
    assert out["original_reply"] == out["previous_output"] == original
    assert out["format_repair"] == repair["format_repair"]
    assert json.loads(normalize(json.dumps(repair))) == out
    assert normalize(normalize(text)) == normalize(text)


def test_thin_bbox_stays_valid(package):
    normalize = importlib.import_module(package + ".coordinate_precision").normalize_prompt
    box = normalize({"bbox": [10.011, 0.011, 10.012, 0.012]})["bbox"]
    assert box == [10.0, 0.0, 10.1, 0.1]
    assert normalize({"bbox": box})["bbox"] == box


@pytest.mark.parametrize("as_text", [False, True])
def test_identity_saved_prompt_and_wire_agree_without_reusing_old_result(package, tmp_path, monkeypatch, as_text):
    model_module = importlib.import_module(package + ".model")
    client = importlib.import_module(package + ".client")
    prompt = {"targets": [{"id": "A1", "bbox": [1.234567, 2.345678, 30.456789, 40.567891]}]}
    if as_text:
        prompt = "本批候选：" + json.dumps(prompt)
    before = copy.deepcopy(prompt)
    config = dict(base_url="http://offline.invalid", model="offline", temperature=0.1,
                  max_tokens=113000, thinking={"type": "enabled"}, api_concurrency=1,
                  format_retries=0, transport_retries=0)
    folder = tmp_path / "request"
    folder.mkdir()
    old = dict(system="SYSTEM", prompt=prompt, images=[], inference={
        k: config[k] for k in ("base_url", "model", "temperature", "max_tokens", "thinking")})
    (folder / "identity.json").write_text(json.dumps(old))
    (folder / "result.json").write_text('{"old_result":true}')
    calls = []

    def offline_request(config, system, actual, images, path):
        payload = client.build_payload(config, system, actual, images)
        path.mkdir(parents=True)
        (path / "wire_request.json").write_text(json.dumps(payload))
        calls.append(path)
        return '{"regions": [], "untouched_output": 0.123456789}'

    monkeypatch.setattr(model_module, "request", offline_request)
    model = model_module.Model(config, system="SYSTEM")
    result = model.ask(folder, prompt, [], lambda x: x)
    assert len(calls) == 1 and result["untouched_output"] == 0.123456789
    active = Path(json.loads((folder / "active_request.json").read_text())["path"])
    assert active.parent == folder / "revisions"
    identity = json.loads((active / "identity.json").read_text())
    saved = json.loads((active / "prompt.json").read_text())["prompt"]
    wire = json.loads((calls[0] / "wire_request.json").read_text())
    wire_text = wire["messages"][1]["content"][0]["text"]
    assert identity["prompt"] == saved
    assert wire_text == (saved if isinstance(saved, str) else json.dumps(saved, ensure_ascii=False))
    assert prompt == before
    assert json.loads((folder / "identity.json").read_text()) == old
    assert wire["thinking"] == config["thinking"] and "reasoning_effort" not in wire
    assert model.ask(folder, prompt, [], lambda x: x) == result
    assert len(calls) == 1

