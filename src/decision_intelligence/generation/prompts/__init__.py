"""Versioned prompt templates.

Each prompt lives in a Markdown file named ``<name>.md``. The file name carries
the version (``_v1``), and the SHA-256 of the content is recorded in the
experiment manifest so that any edit after freezing is detected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    sha256: str

    @property
    def version(self) -> str:
        return self.name


def load_prompt(name: str) -> Prompt:
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    return Prompt(name=name, system=text.strip(), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def prompt_names() -> list[str]:
    return sorted(p.name[:-3] for p in resources.files(__package__).iterdir() if p.name.endswith(".md"))
