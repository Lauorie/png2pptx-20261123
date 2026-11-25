"""Optional reasoning effort and output cap for the arrow branch only.

Arrow geometry sends one small request per candidate region, so a run-wide
"high" effort is spent hundreds of times on questions that need little thought.
These settings let a run keep "high" elsewhere and bound the arrow branch.
Nothing changes unless they are set.
"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

EFFORTS = ('low', 'medium', 'high')
ARROW_ROUTE_PREFIX = '/arrow'


@dataclass(frozen=True)
class ArrowBudget:
    """Per-request limits for the arrow branch; ``None`` keeps the default."""

    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None and self.reasoning_effort not in EFFORTS:
            raise ValueError('Invalid arrow reasoning effort: %r' % (self.reasoning_effort,))
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('Arrow max tokens must be positive')


def to_settings(settings: Mapping[str, Any], effort: Optional[str],
                max_tokens: Optional[int]) -> Dict[str, Any]:
    """Return run settings with the arrow budget recorded.

    Args:
        settings: Existing run settings; not modified.
        effort: Arrow reasoning effort, or None to keep the run-wide behaviour.
        max_tokens: Arrow output cap, or None to keep the backend's own cap.

    Returns:
        A new settings dictionary.

    Raises:
        ValueError: When the effort or cap is invalid.
    """
    budget = ArrowBudget(effort, max_tokens)
    result = dict(settings)
    if budget.reasoning_effort is not None:
        result['arrow_reasoning_effort'] = budget.reasoning_effort
    if budget.max_tokens is not None:
        result['arrow_max_tokens'] = budget.max_tokens
    return result


def from_settings(settings: Mapping[str, Any]) -> ArrowBudget:
    """Read the budget recorded by :func:`to_settings`."""
    return ArrowBudget(settings.get('arrow_reasoning_effort'), settings.get('arrow_max_tokens'))


def arrow_base_url(base_url: str) -> str:
    """Return the gateway address the arrow branch uses, so its requests can be told apart.

    Raises:
        ValueError: When the gateway address does not end in ``/v1``.
    """
    base = base_url.rstrip('/')
    if not base.endswith('/v1'):
        raise ValueError('Gateway base URL must end in /v1: ' + base_url)
    return base[:-len('/v1')] + ARROW_ROUTE_PREFIX + '/v1'


def routed_body(path: str, body: Mapping[str, Any], settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the request body the gateway should relay for a chat route.

    Args:
        path: HTTP request path.
        body: Decoded request body; not modified.
        settings: Run settings.

    Returns:
        The body to relay, or None when the path is not a chat-completions route.
    """
    if path == '/v1/chat/completions':
        return dict(body)
    if path == ARROW_ROUTE_PREFIX + '/v1/chat/completions':
        return bounded(body, from_settings(settings))
    return None


def bounded(payload: Mapping[str, Any], budget: ArrowBudget) -> Dict[str, Any]:
    """Apply the budget to one arrow request received by the local gateway.

    The effort is marked as pinned so the gateway's run-wide forced effort does
    not replace it.  The output cap is only ever lowered.

    Args:
        payload: Request payload carrying ``chat_template_kwargs``; not modified.
        budget: Limits to apply.

    Returns:
        A new payload.
    """
    result = copy.deepcopy(dict(payload))
    if budget.reasoning_effort is not None:
        template = result.setdefault('chat_template_kwargs', {})
        template['reasoning_effort'] = budget.reasoning_effort
        template['pin_reasoning_effort'] = True
    if budget.max_tokens is not None and result.get('max_tokens') is not None:
        result['max_tokens'] = min(result['max_tokens'], budget.max_tokens)
    return result
