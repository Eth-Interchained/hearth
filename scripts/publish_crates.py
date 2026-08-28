#!/usr/bin/env python3
"""Publish every publishable workspace crate to crates.io, in dependency order.

    python3 scripts/publish_crates.py           # publish
    python3 scripts/publish_crates.py --dry-run # show the order and stop

WHY THIS IS DERIVED AND NOT A LIST

The release workflow used to name four crates in a shell script:

    cargo publish -p hearth-core   || echo "already published"
    cargo publish -p hearth-resolve || echo "already published"
    cargo publish -p hearth-store  || echo "already published"
    cargo publish -p hearth-serve  || echo "already published"

Adding `hearth-pull` broke that in two ways at once, and neither would have
been visible in a green CI run:

  1. hearth-pull was not in the list, so it would never publish.
  2. hearth-serve depends on hearth-pull. Publishing hearth-serve against a
     version of hearth-pull that is not on crates.io FAILS — and the
     `|| echo "already published"` swallowed it, so the job would have gone
     green while the flagship crate silently did not ship.

That is the same masked-failure pattern that `|| true` created in the npm job,
and the same allow-list-drift that made scripts/bump.py miss four files twice.
So: read the workspace, topologically sort it, and treat an unexpected failure
as fatal. A new crate is covered the day it is added, by nobody having to
remember anything.

WHAT COUNTS AS OK

"Already published" is genuinely fine — a crate may have gone out on an earlier
tag while another registry failed, and re-running a release must not be blocked
by that. Anything else is fatal. The distinction is made by reading what cargo
actually said, not by ignoring the exit code.
"""

import json
import subprocess
import sys

# Binding crates. They are published to npm and PyPI as native modules; their
# Rust source is not useful standalone on crates.io.
SKIP = {"hearth-node", "hearth-py"}

# Phrases that mean "this exact version is already on crates.io".
ALREADY = (
    "already exists",
    "already uploaded",
    "crate version is already being uploaded",
)


def workspace_packages():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(out.stdout)
    pkgs = {}
    for p in meta["packages"]:
        if p["name"] in SKIP:
            continue
        # `publish = []` or `publish = false` means do not publish.
        if p.get("publish") == []:
            continue
        pkgs[p["name"]] = {
            d["name"]
            for d in p["dependencies"]
            if d["name"] in {q["name"] for q in meta["packages"]}
        }
    # Drop skipped crates from dependency sets so they cannot affect ordering.
    for name in pkgs:
        pkgs[name] -= SKIP
    return pkgs


def publish_order(pkgs):
    """Topological sort. Deterministic — ties broken by name, so the order in a
    log is reproducible and a diff between two releases means something."""
    done, order = set(), []
    while len(order) < len(pkgs):
        ready = sorted(n for n, deps in pkgs.items() if n not in done and deps <= done)
        if not ready:
            stuck = sorted(set(pkgs) - done)
            raise SystemExit(
                f"cannot order these crates — a dependency cycle, or a dep on "
                f"something outside the workspace: {stuck}"
            )
        for n in ready:
            order.append(n)
            done.add(n)
    return order


def publish(name, dry_run):
    if dry_run:
        print(f"  would publish {name}")
        return True
    print(f"\n=== publishing {name} ===", flush=True)
    # cargo waits for the index itself before returning, which is what makes
    # publishing a dependent crate immediately afterwards safe.
    r = subprocess.run(
        ["cargo", "publish", "-p", name, "--allow-dirty"],
        capture_output=True,
        text=True,
    )
    said = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        print(f"  {name}: published")
        return True
    if any(phrase in said.lower() for phrase in ALREADY):
        print(f"  {name}: already on crates.io at this version — fine")
        return True
    # Anything else is a real failure and must stop the release rather than
    # being echoed past.
    print(said, file=sys.stderr)
    print(f"\n  {name}: FAILED — not continuing", file=sys.stderr)
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    pkgs = workspace_packages()
    order = publish_order(pkgs)

    print("publish order (derived from the workspace, not a list):")
    for i, n in enumerate(order, 1):
        deps = ", ".join(sorted(pkgs[n])) or "—"
        print(f"  {i}. {n:<16} after: {deps}")
    print()

    for name in order:
        if not publish(name, dry_run):
            raise SystemExit(1)

    print(f"\nall {len(order)} crate(s) accounted for")


if __name__ == "__main__":
    main()
