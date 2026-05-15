"""Diff main-list packages between two pip-freeze lockfiles.

Usage: compute_bumps.py <env-yaml> <old-lockfile> <new-lockfile>

Main-list packages are taken from the pip section of <env-yaml> (excluding
flags and URLs). Prints one bullet per changed main-list package to stdout:

  - numpy: 2.10.0 -> 2.11.0
  - torch: (added) -> 2.12.0
  - foo:   1.0.0 -> (removed)

Prints nothing if no main-list package changed. Missing old lockfile is
treated as empty (i.e. initial run).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main_packages(yaml_path: Path) -> set[str]:
    spec = yaml.safe_load(yaml_path.read_text())
    names: set[str] = set()
    for dep in spec.get("dependencies", []):
        if not (isinstance(dep, dict) and "pip" in dep):
            continue
        for item in dep["pip"]:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s or s.startswith("-") or "://" in s:
                continue
            name = re.split(r"[<>=!~\s\[]", s, 1)[0]
            if name:
                names.add(norm(name))
    return names


def parse_lock(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        name, ver = line.split("==", 1)
        ver = ver.split(";")[0].strip()
        out[norm(name)] = ver
    return out


def main() -> int:
    if len(sys.argv) != 4:
        sys.stderr.write(f"Usage: {sys.argv[0]} <yaml> <old-lock> <new-lock>\n")
        return 2
    yaml_path, old_path, new_path = (Path(p) for p in sys.argv[1:])
    mains = main_packages(yaml_path)
    old = parse_lock(old_path)
    new = parse_lock(new_path)

    lines: list[str] = []
    for pkg in sorted(mains):
        ov, nv = old.get(pkg), new.get(pkg)
        if ov == nv:
            continue
        if ov is None:
            lines.append(f"  - {pkg}: (added) -> {nv}")
        elif nv is None:
            lines.append(f"  - {pkg}: {ov} -> (removed)")
        else:
            lines.append(f"  - {pkg}: {ov} -> {nv}")

    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
