#!/usr/bin/env python3
"""Bump every version-bearing file at once.

Three files carry the version and they must never drift: a Cargo workspace
that says 0.1.1 while package.json says 0.1.0 produces a release where the
npm package and the crate are different software wearing the same number.
NEDB learned this the hard way; hearth gets the tool on day one.

    python3 scripts/bump.py 0.1.1
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

FILES = [
    # (path, regex with ONE capture group around the version, description)
    ("Cargo.toml", r'(?m)^(version = ")[^"]+(")', "cargo workspace"),
    ("crates/hearth-node/package.json", r'("version":\s*")[^"]+(")', "npm"),
    ("crates/hearth-py/pyproject.toml", r'(?m)^(version = ")[^"]+(")', "pypi"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    new = sys.argv[1].lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        print(f"not a semver: {new}")
        return 2

    for rel, pattern, what in FILES:
        path = ROOT / rel
        text = path.read_text()
        updated, n = re.subn(pattern, lambda m: f"{m.group(1)}{new}{m.group(2)}", text, count=1)
        if n != 1:
            print(f"FAILED to find a version in {rel} — refusing to ship a partial bump")
            return 1
        path.write_text(updated)
        print(f"  {what:16} {rel} -> {new}")

    print(f"\nall three at {new}. next:")
    print(f"  cargo test -p hearth-core && git commit -am 'chore: v{new}' "
          f"&& git tag -a v{new} -m ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
