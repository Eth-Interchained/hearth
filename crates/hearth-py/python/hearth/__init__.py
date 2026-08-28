"""hearth — deterministic model residency.

The Rust core hands JSON across the boundary and this module turns it into
dicts. Thin on purpose: every rule about residency and VRAM lives in
hearth-core and is tested there once, in Rust. A binding that reimplements any
of it becomes a second source of truth, and second sources of truth disagree
with the first one eventually.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping

from . import _hearth  # type: ignore[attr-defined]

__version__ = _hearth.__version__
GIB = 1024 ** 3


def now_ms() -> int:
    return int(time.time() * 1000)


def declare(model: str, weights_bytes: int, kv_bytes: int = 0) -> dict[str, Any]:
    """One model you intend to keep resident, and what you expect it to cost."""
    return {"model": model, "weights_bytes": weights_bytes, "kv_bytes": kv_bytes}


def plan(total_bytes: int, declared: Iterable[Mapping[str, Any]],
         reserve_pct: int = 8) -> dict[str, Any]:
    """Will this card hold this roster? Answered before anything loads.

    The reserve is never planned into: KV cache grows with context and
    parallelism, the CUDA context costs hundreds of megabytes, and
    fragmentation is real on a card that has been up for weeks.
    """
    return json.loads(_hearth.plan(int(total_bytes), int(reserve_pct),
                                   json.dumps(list(declared))))


class Fleet:
    """Every declared model on one host, in priority order.

    Declaration order IS priority order — first fit, never best fit, because
    reordering to squeeze one more model in silently demotes whatever you
    listed first.
    """

    def __init__(self, total_bytes: int, declared: Iterable[Mapping[str, Any]],
                 reserve_pct: int = 8) -> None:
        self._f = _hearth.Fleet(int(total_bytes), int(reserve_pct),
                                json.dumps(list(declared)))

    def set_endpoint(self, model: str, endpoint: str) -> None:
        self._f.set_endpoint(model, endpoint)

    def observe(self, model: str, kind: str, detail: Mapping[str, Any] | None = None,
                now: int | None = None) -> None:
        """Record a fact. Never a conclusion.

        kind: load_started · probe_ok · probe_failed · process_exited ·
              load_failed · stop

        On probe_failed, `gpu_present` is the most important field you will
        ever pass here: it is the whole difference between the runtime dropping
        a model and the host taking the card away. Omitting it is read as
        "still present", so a missing field can never quietly exonerate an
        operator.
        """
        self._f.observe(model, kind, json.dumps(dict(detail or {})),
                        now_ms() if now is None else int(now))

    def route(self, model: str, now: int | None = None) -> dict[str, Any]:
        """What a router should do with a request for this model."""
        return json.loads(self._f.route(model, now_ms() if now is None else int(now)))

    def next_to_load(self) -> str | None:
        """The next model worth bringing up, or None when everything that can
        be warm already is. Nothing is ever evicted to make room."""
        return self._f.next_to_load()

    @property
    def committed_bytes(self) -> int:
        return self._f.committed_bytes()

    @property
    def free_bytes(self) -> int:
        return self._f.free_bytes()

    def report(self, now: int | None = None) -> str:
        return self._f.report(now_ms() if now is None else int(now))


__all__ = ["Fleet", "plan", "declare", "now_ms", "GIB", "__version__"]
