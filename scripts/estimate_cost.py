#!/usr/bin/env python3
"""Estimate the model cost of one finished run from its recorded token usage.

Chat usage comes from RUN_DIR/api_usage.jsonl (written by the worker). Image-edit usage comes
from the image adapter's usage log, restricted to the run's time window, so it is only exact
when one run used the adapter during that window.
"""
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the USD cost of the given token counts."""
        return (prompt_tokens * self.input_per_million + completion_tokens * self.output_per_million) / 1e6


def _rows(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _summarise(rows: Iterator[Dict[str, Any]], price: Price) -> Dict[str, Any]:
    calls = prompt = completion = 0
    for row in rows:
        usage = row.get("usage") or {}
        calls += 1
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return {"calls": calls, "prompt_tokens": prompt, "completion_tokens": completion,
            "usd": round(price.cost(prompt, completion), 4)}


def estimate(run_dir: Path, chat_price: Price, image_price: Price,
             image_usage_log: Optional[Path]) -> Dict[str, Any]:
    """Compute the cost summary of one run.

    Args:
        run_dir: Finished run directory.
        chat_price: Price of the chat model.
        image_price: Price of the image model.
        image_usage_log: Usage log of the image adapter, if any.

    Returns:
        A JSON-serialisable summary.

    Raises:
        FileNotFoundError: When the run has no usage or status file.
    """
    chat_rows = list(_rows(run_dir / "api_usage.jsonl"))
    status = json.loads((run_dir / "status.json").read_text())
    images = int(status.get("exported") or status.get("completed") or 0)
    start = min((row["at"] - row.get("elapsed_seconds", 0) for row in chat_rows), default=0.0)
    end = float(status.get("at") or 0.0)
    result: Dict[str, Any] = {"run_dir": str(run_dir), "state": status.get("state"), "images": images,
                              "chat": _summarise(iter(chat_rows), chat_price)}
    result["chat"]["models"] = sorted({str(row.get("returned_model") or row.get("model")) for row in chat_rows})
    if image_usage_log is not None:
        window = (row for row in _rows(image_usage_log) if start <= row.get("at", 0) <= end)
        result["image_edit"] = _summarise(window, image_price)
    total = result["chat"]["usd"] + result.get("image_edit", {}).get("usd", 0.0)
    result["total_usd"] = round(total, 4)
    result["usd_per_image"] = round(total / images, 4) if images else None
    return result


def main() -> None:
    """Print the cost summary of a run as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--image-usage-log", type=Path)
    parser.add_argument("--chat-input-price", type=float, required=True, help="USD per million input tokens")
    parser.add_argument("--chat-output-price", type=float, required=True, help="USD per million output tokens")
    parser.add_argument("--image-input-price", type=float, default=0.0)
    parser.add_argument("--image-output-price", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = estimate(args.run_dir, Price(args.chat_input_price, args.chat_output_price),
                       Price(args.image_input_price, args.image_output_price), args.image_usage_log)
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
