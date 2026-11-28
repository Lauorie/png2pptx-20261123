"""Bounded transport retries, separate from geometry/schema revision budgets."""
import http.client
import json
import time
import urllib.error
import urllib.request
from core import save


class ModelTransportError(RuntimeError):
    """No usable service response; never feed this failure to the model as geometry feedback."""


class ModelRefusedError(ModelTransportError):
    """The provider rejected this very request (HTTP 422); the same request cannot succeed later."""


def request_json(opener, url, body, *, trace_path=None, timeout=1200., attempts=3, headers=None, busy_only=False):
    if timeout <= 0 or attempts < 1:
        raise ValueError('Positive transport timeout and attempt count required')
    # Headers are deliberately absent from request/transport traces.
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', **(headers or {})})
    started = time.monotonic()
    history = []
    for attempt in range(attempts):
        remaining = timeout-(time.monotonic()-started)
        if remaining <= 0:
            break
        entry = {'attempt': attempt, 'started_at': time.time()}
        delay = min(8., 2.**attempt)
        try:
            with opener.open(request, timeout=remaining) as response:
                result = json.load(response)
            entry.update(status='received', seconds=time.monotonic()-started)
            history.append(entry)
            if trace_path:
                save(trace_path, {'attempts': history, 'status': 'received'})
            return result
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if busy_only:
                # This exact service response precedes acquiring its inference
                # lock. The request was NOT accepted, so retry cannot create a
                # second writer. Timeouts and all other errors remain terminal.
                try: detail=json.loads(exc.read(4096))
                except (ValueError,UnicodeError): detail={}
                retryable=(exc.code==503 and detail.get('error')=='SAM is busy; retry after this request finishes')
            entry.update(error_type=type(exc).__name__, http_status=exc.code, retryable=retryable)
            retry_after = exc.headers.get('Retry-After') if exc.headers else None
            try:
                if retry_after is not None:
                    delay = min(30., max(delay, float(retry_after)))
            except ValueError:
                pass
            exc.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            retryable = not busy_only
            entry.update(error_type=type(exc).__name__, retryable=retryable)
        except (ValueError, UnicodeError) as exc:
            retryable = False
            entry.update(error_type=type(exc).__name__, retryable=False)
        entry.update(status='failed', seconds=time.monotonic()-started)
        history.append(entry)
        if trace_path:
            save(trace_path, {'attempts': history, 'status': 'failed'})
        remaining = timeout-(time.monotonic()-started)
        if not retryable or attempt+1 >= attempts or remaining <= delay:
            break
        time.sleep(delay)
    last = history[-1] if history else {'error_type': 'DeadlineExceeded'}
    if last.get('http_status') == 422:
        raise ModelRefusedError('Model service refused the request (HTTP 422); output unavailable')
    raise ModelTransportError('Model transport failed after '+str(len(history))+' attempt(s): '+
                              last['error_type']+(' HTTP '+str(last['http_status']) if 'http_status' in last else ''))
