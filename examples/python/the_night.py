"""The night hearth was built for, replayed in Python.

    python examples/python/the_night.py

A model warms up, serves for an hour, and then the host takes the card away.
Watch what a router is told at each step -- and in particular, who gets blamed
at the end.

Install:  pip install hearth-engine
From a checkout:  cd crates/hearth-py && maturin develop
"""

import json

from hearth import GIB, Fleet, declare, plan

T0 = 1_756_000_000_000  # a fixed instant, so the output is stable
SEC = 1_000


def show(label: str, fleet: Fleet, model: str, now: int) -> None:
    r = fleet.route(model, now)
    flags = " ".join(
        f
        for f in (
            "READY" if r.get("ready") else None,
            "try-elsewhere" if r.get("try_elsewhere") else None,
            "OPERATOR-FAULT" if r.get("operator_fault") else None,
        )
        if f
    )
    print(f"{label:<24} {model:<20} {flags or '—'}")
    print(f"{'':<24} {json.dumps(r, sort_keys=True)}")
    print()


def main() -> None:
    print("=== will this card hold this roster? ===\n")

    # The roster that started hearth: five models an operator wanted resident
    # on one rented RTX A6000. No runtime refuses this -- it loads, evicts,
    # loads, evicts, and presents to everyone as "the models got slow".
    roster = [
        declare("muse-local:latest", 20 * GIB, GIB),
        declare("deepseek-r1:32b", 20 * GIB, GIB),
        declare("gemma4:26b", 16 * GIB, GIB),
        declare("qwen3.6:27b", 17 * GIB, GIB),
        declare("gemma4-extract:31b", 19 * GIB, GIB),
    ]

    p = plan(48 * GIB, roster)
    print(p["explain"])
    print(f"declared: {p['declared']}  admitted: {len(p['admitted'])}  rejected: {len(p['rejected'])}")
    print()

    print("=== the night, replayed ===\n")

    fleet = Fleet(48 * GIB, [
        declare("muse-local:latest", 20 * GIB, GIB),
        # Declared but too big for what is left. Recorded, not dropped.
        declare("gemma4:26b", 40 * GIB, GIB),
    ])

    # 1. Nothing observed yet. An honest "I do not know" beats a confident
    #    wrong answer in either direction.
    show("never probed", fleet, "muse-local:latest", T0)

    # 2. Weights materializing. Not ready -- and not a fault.
    fleet.observe("muse-local:latest", "load_started", {}, T0)
    show("loading", fleet, "muse-local:latest", T0 + 20 * SEC)

    # 3. Loaded, answering, accounted for.
    fleet.set_endpoint("muse-local:latest", "127.0.0.1:8090")
    fleet.observe("muse-local:latest", "probe_ok", {"vram_bytes": 21 * GIB}, T0 + 40 * SEC)
    show("resident", fleet, "muse-local:latest", T0 + 40 * SEC)

    # 4. An hour later the probe fails -- and the card is GONE. This is the
    #    whole reason hearth exists. gpu_present is the difference between
    #    "this operator over-committed" and "their provider reclaimed it".
    fleet.observe(
        "muse-local:latest",
        "probe_failed",
        {"gpu_present": False, "detail": "no CUDA device"},
        T0 + 3600 * SEC,
    )
    show("GPU detached", fleet, "muse-local:latest", T0 + 3600 * SEC)

    # The same failure with the card still there is a DIFFERENT diagnosis,
    # and that one IS the operator's to answer for.
    other = Fleet(48 * GIB, [declare("muse-local:latest", 20 * GIB, GIB)])
    other.observe("muse-local:latest", "load_started", {}, T0)
    other.observe("muse-local:latest", "probe_ok", {"vram_bytes": 21 * GIB}, T0 + SEC)
    other.observe(
        "muse-local:latest",
        "probe_failed",
        {"gpu_present": True, "detail": "model not loaded"},
        T0 + 100 * SEC,
    )
    evicted = other.route("muse-local:latest", T0 + 100 * SEC)
    print("same probe failure, card still PRESENT:")
    print(f"{'':<24} {json.dumps(evicted, sort_keys=True)}")
    print(f"{'':<24} operatorFault: {evicted.get('operator_fault')}  <- over-committing IS theirs")
    print()

    # 5. The model that never fit. Permanent until the hardware or the
    #    declaration changes, so a caller should stop asking.
    show("never fit", fleet, "gemma4:26b", T0 + 3600 * SEC)

    print("=== what should be brought up next ===")
    # Never by evicting. If it does not fit, the honest answer is that it does not.
    print(f"next_to_load: {fleet.next_to_load()}")
    print()
    print(fleet.report(T0 + 3600 * SEC))


if __name__ == "__main__":
    main()
