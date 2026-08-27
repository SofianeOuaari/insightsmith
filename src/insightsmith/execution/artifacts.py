"""Where figures and tables are written, and how they are named.

An artifact is a file plus enough context to say where it came from. Every one
records the question that produced it and the hash of the card it was reasoned
from, so a chart found on disk months later can be traced back to the data and
the plan behind it — §11's auditability point, made concrete.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

__all__ = ["Artifact", "ArtifactStore"]

_SAFE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG: Final = 48


@dataclass(slots=True)
class Artifact:
    """One saved file and its provenance."""

    path: Path
    kind: str
    title: str = ""
    question: str = ""
    card_hash: str = ""
    code: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass(slots=True)
class ArtifactStore:
    """A directory of results, with a manifest.

    Writes are additive: a repeated question gets a suffixed name rather than
    silently overwriting an earlier answer.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def write_bytes(self, name: str, payload: bytes, **meta: Any) -> Artifact:
        return self._write(name, payload, **meta)

    def write_text(self, name: str, payload: str, **meta: Any) -> Artifact:
        return self._write(name, payload.encode("utf-8"), **meta)

    def _write(self, name: str, payload: bytes, **meta: Any) -> Artifact:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._free_path(name)
        path.write_bytes(payload)
        artifact = Artifact(path=path, kind=path.suffix.lstrip("."), **meta)
        self._append(artifact)
        return artifact

    def _free_path(self, name: str) -> Path:
        stem, _, suffix = name.rpartition(".")
        stem = slugify(stem or name)
        candidate = self.root / f"{stem}.{suffix}"
        counter = 2
        while candidate.exists():
            candidate = self.root / f"{stem}-{counter}.{suffix}"
            counter += 1
        return candidate

    def _append(self, artifact: Artifact) -> None:
        entries = self.entries()
        entries.append(artifact.to_dict())
        self.manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def entries(self) -> list[dict[str, Any]]:
        """Everything written so far, oldest first."""
        if not self.manifest_path.is_file():
            return []
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            return []
        return loaded if isinstance(loaded, list) else []


def slugify(text: str, *, limit: int = _MAX_SLUG) -> str:
    """A filename that survives every filesystem, from arbitrary prose."""
    cleaned = _SAFE.sub("-", text.lower()).strip("-")
    return cleaned[:limit].rstrip("-") or "artifact"
