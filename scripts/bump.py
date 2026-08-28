#!/usr/bin/env python3
"""Bump every version string in the repo, everywhere it appears.

    python3 scripts/bump.py 0.2.0

WHY THIS IS A SWEEP AND NOT A LIST OF FIELDS

The first version of this script knew about three files. It missed the four
generated npm platform packages, so the main package published
optionalDependencies pointing at versions that did not exist. I added those
four, and it then missed the internal `hearth-core = { version = "0.1.0" }`
pins inside the two binding crates, which broke the build outright.

Twice is a pattern. The problem was never which files — it is that an
allow-list of known places will always be one behind the repo. So this reads
the CURRENT version out of the workspace manifest and replaces that exact
string wherever it appears as a version in any manifest. A new crate, a new
binding, a new platform package: covered the day it is added, by nobody having
to remember anything.

It refuses to finish if any manifest still mentions the old version, because a
partial bump that reports success is worse than a failure.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Anywhere a version can hide. Globs, not names, so new crates are covered.
MANIFEST_GLOBS = [
    "Cargo.toml",
    "crates/*/Cargo.toml",
    "crates/*/pyproject.toml",
    "crates/*/package.json",
    "crates/*/npm/*/package.json",
]


def manifests():
    out = []
    for pattern in MANIFEST_GLOBS:
        out.extend(sorted(ROOT.glob(pattern)))
    # target/ and node_modules/ hold copies that are not ours to edit.
    return [p for p in out if "target" not in p.parts and "node_modules" not in p.parts]


def current_version():
    root = (ROOT / "Cargo.toml").read_text()
    m = re.search(r'(?m)^version = "([^"]+)"', root)
    if not m:
        raise SystemExit("no version in the workspace Cargo.toml — refusing to guess")
    return m.group(1)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    new = sys.argv[1].lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        print(f"not a semver: {new}")
        return 2

    old = current_version()
    if old == new:
        print(f"already at {new}")
        return 0

    # Only where the old version is used AS a version. Never a bare substring,
    # which would cheerfully rewrite a URL or a digest containing those digits.
    patterns = [
        re.compile(r'(version = ")' + re.escape(old) + r'(")'),
        re.compile(r'("version":\s*")' + re.escape(old) + r'(")'),
    ]

    touched = 0
    for path in manifests():
        original = path.read_text()
        text = original
        hits = 0
        for pattern in patterns:
            text, n = pattern.subn(lambda m: f"{m.group(1)}{new}{m.group(2)}", text)
            hits += n
        if text != original:
            path.write_text(text)
            touched += 1
            print(f"  {hits:>2}x  {path.relative_to(ROOT)}")

    if touched == 0:
        print(f"found nothing at {old} to bump — has the tree already moved?")
        return 1

    # Prove it. No manifest may still mention the old version anywhere.
    stragglers = [
        str(p.relative_to(ROOT)) for p in manifests() if f'"{old}"' in p.read_text()
    ]
    if stragglers:
        print(f"\nSTILL AT {old}: {', '.join(stragglers)}")
        print("refusing to call this a bump")
        return 1

    print(f"\n{old} -> {new} across {touched} file(s), no stragglers")
    print(f"  cargo test && git commit -am 'chore: v{new}' && git tag -a v{new} -m ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
