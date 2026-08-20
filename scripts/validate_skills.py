#!/usr/bin/env python3
"""Validate the structure and public hygiene of every skill in this repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")
ABSOLUTE_PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)")


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening YAML frontmatter delimiter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("missing closing YAML frontmatter delimiter") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"unsupported frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, text


def validate_skill(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        return ["missing SKILL.md"]

    try:
        metadata, text = parse_frontmatter(skill_file)
    except ValueError as exc:
        return [str(exc)]

    name = metadata.get("name", "")
    description = metadata.get("description", "")
    if not NAME_RE.fullmatch(name):
        errors.append("name must contain lowercase letters, numbers, and hyphens only")
    if name != skill_dir.name:
        errors.append(f"frontmatter name {name!r} does not match directory {skill_dir.name!r}")
    if not description:
        errors.append("description is required")
    elif len(description) > 1024:
        errors.append("description exceeds 1,024 characters")
    if len(text.splitlines()) > 500:
        errors.append("SKILL.md exceeds the 500-line progressive-disclosure target")
    if ABSOLUTE_PATH_RE.search(text):
        errors.append("contains a machine-specific absolute path")

    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean_target = target.split("#", 1)[0]
        if clean_target and not (skill_dir / clean_target).resolve().exists():
            errors.append(f"broken local Markdown link: {target}")
    return errors


def main() -> int:
    if not SKILLS.is_dir():
        print("ERROR: skills/ directory is missing")
        return 1

    failures = 0
    for skill_dir in sorted(path for path in SKILLS.iterdir() if path.is_dir()):
        errors = validate_skill(skill_dir)
        if errors:
            failures += 1
            print(f"FAIL {skill_dir.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {skill_dir.name}")

    if failures:
        print(f"\n{failures} skill(s) failed validation")
        return 1
    print("\nAll skills passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
