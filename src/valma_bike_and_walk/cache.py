"""Knowing when a cached file is still the answer.

Everything under ``--cache-dir`` is derived: a link layer extracted from a PBF,
a graph built from a link layer. A derived file is only worth reusing while
what it was derived *from* is unchanged -- and in this project the inputs
change all the time, by design: the link layer is meant to be edited in QGIS,
and ``valma dem`` and ``valma traffic`` write their columns back into it in
place. A cache keyed on a file *name* would keep serving the graph from before
the edit, with no sign anything was wrong.

So every cached file carries a **key**: a small JSON-able dict saying what it
was built from. Before a cache is reused its stored key is compared with the key
the current inputs would produce, and any difference means rebuild. A key holds:

- a stamp of every input file -- resolved path, size and modification time
  (:func:`file_stamp`). That is what ``make`` trusts, and it is cheap on a
  gigabyte GeoPackage where hashing the content would not be;
- a digest of the source code that does the building (:func:`code_digest`), so
  a change to a speed profile or a penalty rebuilds what it affects rather than
  hiding behind an old graph;
- whatever else selects the output: the mode, the extent.

Where the key is stored is up to the file type: a network ``.npz`` carries its
own (see :class:`valma_bike_and_walk.network.RoutableNetwork`), anything else
gets a JSON sidecar beside it (:func:`read_sidecar`, :func:`write_sidecar`).
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

#: Bump to invalidate every existing cache when the key layout itself changes.
KEY_FORMAT = 1

#: SQLite -- and so GeoPackage -- side files. QGIS edits a GeoPackage in WAL
#: mode, where committed edits can sit in ``<file>-wal`` with the main file's
#: own size and mtime untouched until a checkpoint. GDAL reads through the WAL,
#: so it is part of what a build sees and has to be part of the stamp.
_SQLITE_SIDE_FILES = ("-wal",)

_SIDECAR_SUFFIX = ".key.json"


def _stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def file_stamp(path: Path | str) -> dict[str, Any]:
    """
    Identity of an input file as it is right now: path, size and mtime.

    Take the stamp *before* reading the file, so an edit saved while a build is
    reading it leaves the result looking stale rather than falsely fresh.
    """
    path = Path(path).resolve()
    stamp: dict[str, Any] = {"path": str(path), **_stat(path)}
    for suffix in _SQLITE_SIDE_FILES:
        side = path.with_name(path.name + suffix)
        if side.exists():
            stamp[suffix] = _stat(side)
    return stamp


def path_digest(path: Path | str, length: int = 8) -> str:
    """Short, stable digest of a file's resolved path, for naming its caches."""
    resolved = str(Path(path).resolve()).encode("utf-8")
    return hashlib.blake2b(resolved, digest_size=16).hexdigest()[:length]


@lru_cache(maxsize=None)
def code_digest(*modules: str) -> str:
    """
    Digest of the source of the named modules.

    A cache built by older code is stale even if its data inputs are not: the
    graph is a function of the link layer *and* of the profiles and penalties
    applied to it. Hashing the source means nobody has to remember to bump a
    version when they change a coefficient. A docstring edit costs one
    unnecessary rebuild, which is the right side to err on.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(modules):
        module = import_module(name)
        digest.update(name.encode("utf-8"))
        source = getattr(module, "__file__", None)
        if source is not None and os.path.exists(source):
            digest.update(Path(source).read_bytes())
        else:  # pragma: no cover - frozen or zipped installs
            from valma_bike_and_walk import __version__

            digest.update(__version__.encode("utf-8"))
    return digest.hexdigest()


def make_key(kind: str, **parts: Any) -> dict[str, Any]:
    """A cache key: what kind of file, and everything it was built from."""
    return {"format": KEY_FORMAT, "kind": kind, **parts}


def dump_key(key: Mapping[str, Any]) -> str:
    return json.dumps(key, sort_keys=True)


def load_key(text: str) -> dict[str, Any] | None:
    """Parse a stored key; anything unreadable counts as no key at all."""
    if not text:
        return None
    try:
        key = json.loads(text)
    except ValueError:
        return None
    return key if isinstance(key, dict) else None


def stale_reason(
    stored: Mapping[str, Any] | None, wanted: Mapping[str, Any]
) -> str | None:
    """
    Why a cache with key ``stored`` cannot stand in for ``wanted``, or None.

    Only the names of the parts that differ are reported -- enough for a log
    line to say "links changed" rather than dumping two stamps side by side.
    """
    if stored is None:
        return "it records no inputs"
    # Round-trip through JSON so tuples and lists compare equal.
    stored = json.loads(dump_key(stored))
    wanted = json.loads(dump_key(wanted))
    changed = sorted(
        name
        for name in set(stored) | set(wanted)
        if stored.get(name) != wanted.get(name)
    )
    if not changed:
        return None
    return ", ".join(changed) + (" has" if len(changed) == 1 else " have") + " changed"


def sidecar_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def read_sidecar(path: Path | str) -> dict[str, Any] | None:
    """The key stored beside ``path``, or None if there is none."""
    side = sidecar_path(path)
    if not side.exists():
        return None
    return load_key(side.read_text(encoding="utf-8"))


def write_sidecar(path: Path | str, key: Mapping[str, Any]) -> None:
    """Record ``key`` beside ``path``. Call only once ``path`` is fully written."""
    sidecar_path(path).write_text(dump_key(key), encoding="utf-8")


def remove_sidecar(path: Path | str) -> None:
    """
    Forget ``path``'s key. Call *before* rewriting ``path``.

    A build that dies halfway through writing must not leave a half-written
    file next to a key that vouches for it.
    """
    sidecar_path(path).unlink(missing_ok=True)
