"""Render GitHub release notes for a torch-env tag.

Usage:
  render_release_notes.py <tag> [--bumps-linux PATH] [--bumps-mac PATH] [--seed]

Reads the current lockfiles and yamls in the working tree (which should
be checked out at the tag) and emits markdown to stdout:

  - "Bumps" section per platform (from the optional bumps files written
    by compute_bumps.py).
  - Versions table: one row per main-list package, columns linux | mac.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main_packages(yaml_path: Path) -> list[str]:
    spec = yaml.safe_load(yaml_path.read_text())
    out: list[str] = []
    for dep in spec.get("dependencies", []):
        if not (isinstance(dep, dict) and "pip" in dep):
            continue
        for item in dep["pip"]:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s or s.startswith("-") or "://" in s:
                continue
            name = re.split(r"[<>=!~\s\[]", s, maxsplit=1)[0]
            if name:
                out.append(norm(name))
    return out


def parse_lock(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-") or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        out[norm(name)] = ver.split(";")[0].strip()
    return out


def render(tag: str, bumps_linux: str, bumps_mac: str, seed: bool) -> str:
    linux_lock = parse_lock(Path("lockfiles/requirements.linux.txt"))
    mac_lock   = parse_lock(Path("lockfiles/requirements.mac.txt"))

    main = sorted(set(main_packages(Path("environment.linux.yaml"))) |
                  set(main_packages(Path("environment.mac.yaml"))))

    lines: list[str] = [f"# torch-env {tag}", ""]

    if seed:
        lines += ["Initial tag.", ""]

    if bumps_linux or bumps_mac:
        lines.append("## Bumps")
        if bumps_linux:
            lines += ["", "**linux**", "", "```", bumps_linux.rstrip(), "```"]
        if bumps_mac:
            lines += ["", "**mac**", "", "```", bumps_mac.rstrip(), "```"]
        lines.append("")

    lines += [
        "## Main package versions",
        "",
        "| package | linux | mac |",
        "| --- | --- | --- |",
    ]
    for pkg in main:
        lv = linux_lock.get(pkg, "—")
        mv = mac_lock.get(pkg, "—")
        lines.append(f"| `{pkg}` | {lv} | {mv} |")

    return "\n".join(lines) + "\n"


def _read(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    return p.read_text() if p.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--bumps-linux", default=None)
    ap.add_argument("--bumps-mac", default=None)
    ap.add_argument("--seed", action="store_true")
    args = ap.parse_args()
    sys.stdout.write(render(
        tag=args.tag,
        bumps_linux=_read(args.bumps_linux),
        bumps_mac=_read(args.bumps_mac),
        seed=args.seed,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
