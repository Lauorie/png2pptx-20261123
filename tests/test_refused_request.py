"""A request the provider refuses is neither retried nor allowed to fail the figure."""
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor'), str(ROOT / 'experiment/shape_sam_guided'),
                str(ROOT / 'experiment/style_line_sam')]

import test_pipeline as fixtures
from request_io import ModelTransportError, request_json
from test_pipeline import FakeRuntime, observed, reviewed


def _error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError('http://gateway/', code, 'failed', {}, io.BytesIO(json.dumps({'error': 'x'}).encode()))


class RefusedRequestTransportTests(unittest.TestCase):
    def test_a_refusal_is_reported_as_its_own_error_after_one_try(self) -> None:
        from request_io import ModelRefusedError
        opener = Mock()
        opener.open.side_effect = [_error(422), io.BytesIO(b'{"ok":true}')]
        with patch('request_io.time.sleep'), self.assertRaises(ModelRefusedError):
            request_json(opener, 'http://gateway/', {})
        self.assertEqual(opener.open.call_count, 1)

    def test_an_outage_is_still_a_plain_transport_failure(self) -> None:
        from request_io import ModelRefusedError
        opener = Mock()
        opener.open.side_effect = [_error(502) for _ in range(3)]
        with patch('request_io.time.sleep'), self.assertRaises(ModelTransportError) as caught:
            request_json(opener, 'http://gateway/', {})
        self.assertNotIsInstance(caught.exception, ModelRefusedError)
        self.assertEqual(opener.open.call_count, 3)


class ModelRefusedError(RuntimeError):
    """Stands in for request_io.ModelRefusedError; the fitting code matches by name."""


class RefusedRequestFittingTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def _run(self, refuse: str) -> Any:
        def respond(tag: str, prompt: Any, count: int) -> Any:
            if tag == refuse:
                raise ModelRefusedError('Model service refused the request')
            return observed('G1') if tag == 'observe' else reviewed(prompt)
        return self.run_object(FakeRuntime(self.config, respond))

    def test_a_refused_review_keeps_the_fitted_geometry_unreviewed(self) -> None:
        record = self._run('check')
        self.assertEqual(record['stop_reason'], 'check_refused')
        self.assertEqual(record['selected_revision'], 'revision_00')

    def test_a_refused_observation_is_not_a_service_outage(self) -> None:
        record = self._run('observe')
        self.assertEqual(record['stop_reason'], 'observation_refused')
