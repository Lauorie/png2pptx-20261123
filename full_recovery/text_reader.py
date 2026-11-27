"""Read a figure's text lines with a vision model behind the OpenAI-compatible gateway.

The reader serves scoring only (see ``text_fidelity.py``): it should be a model
other than the one that recognises text during the conversion, so that the two
do not share their mistakes.
"""
import base64
import io
import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .text_fidelity import compare

logger = logging.getLogger(__name__)

Post = Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
NOTHING = 'NONE'
PROMPT = (
    'Transcribe every piece of text visible in this image exactly as it is written: keep case, punctuation, '
    'digits and symbols, do not correct, complete or translate anything, and write formulas as the plain '
    'characters you see (no LaTeX, no markdown). Include every occurrence, also of text that appears more '
    'than once. Answer with one row per line of text and nothing else. '
    f'If the image shows no text, answer {NOTHING}.'
)


class ReadingFailed(RuntimeError):
    """The model did not return the text lines in the form asked for."""


@dataclass(frozen=True)
class ReaderConfig:
    """Where and how to ask for a reading."""

    base_url: str
    key: str
    model: str
    attempts: int = 3
    max_tokens: int = 16000
    timeout_seconds: float = 300.


def parse_reply(reply: str) -> List[str]:
    """Turn the model's reply, one line of text per row, into the list of lines.

    Raises:
        ReadingFailed: When the reply is empty.
    """
    rows = [row.strip() for row in reply.splitlines()]
    lines = [row for row in rows if row and not row.startswith('```')]
    if not lines:
        raise ReadingFailed('The reply is empty')
    return [] if lines == [NOTHING] else lines


def http_post(url: str, key: str, body: Dict[str, Any], timeout: float = 300.) -> Dict[str, Any]:
    """POST ``body`` as JSON with a bearer key and return the decoded reply."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def make_reader(config: ReaderConfig, post: Post = http_post) -> Callable[[Image.Image], List[str]]:
    """Return a function that reads the text lines of an image with ``config.model``."""

    def read(image: Image.Image) -> List[str]:
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, 'PNG')
        picture = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        body = {'model': config.model, 'temperature': 0, 'max_tokens': config.max_tokens, 'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': PROMPT},
                                         {'type': 'image_url', 'image_url': {'url': picture}}]}]}
        failure: Exception = ReadingFailed('No attempt was made')
        for attempt in range(config.attempts):
            try:
                reply = post(config.base_url.rstrip('/') + '/chat/completions', config.key, body)
                return parse_reply(reply['choices'][0]['message']['content'] or '')
            except (ReadingFailed, KeyError, IndexError, TypeError, urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError) as error:
                logger.warning('Reading attempt %d of %d failed: %s', attempt + 1, config.attempts, error)
                failure = error
        raise ReadingFailed(f'No usable reading after {config.attempts} attempts') from failure

    return read


def consensus(read: Callable[[Image.Image], List[str]],
              scales: Tuple[float, ...] = (1., 1.5, 2.)) -> Callable[[Image.Image], List[str]]:
    """Return a reader that reads a picture at several sizes and keeps the reading the others agree with most.

    A model now and then stops early or wanders; the same picture at another
    size draws an independent reading, and the odd one out is dropped.

    Raises:
        ReadingFailed: From the returned reader, when no size gives a reading.
    """

    def one(image: Image.Image, scale: float) -> Optional[List[str]]:
        size = (round(image.width * scale), round(image.height * scale))
        try:
            return read(image if scale == 1. else image.resize(size, Image.LANCZOS))
        except ReadingFailed as error:
            logger.warning('Reading at scale %.2f failed: %s', scale, error)
            return None

    def read_all(image: Image.Image) -> List[str]:
        picture = image.convert('RGB')
        with ThreadPoolExecutor(max_workers=len(scales)) as pool:
            readings = [r for r in pool.map(lambda scale: one(picture, scale), scales) if r is not None]
        if not readings:
            raise ReadingFailed('No size gave a reading')
        agreement = [sum(compare(mine, other).overall for other in readings) for mine in readings]
        return readings[agreement.index(max(agreement))]

    return read_all
