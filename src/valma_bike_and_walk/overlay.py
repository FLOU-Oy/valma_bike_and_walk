"""Laying this model's travel times over another model's full zone system.

The routed matrix covers only the zones that snapped to the network, and only
the pairs within ``--max-minutes`` of each other. A downstream model wants a
travel time for every pair of *its* zones. So the output starts from a baseline
matrix that already has that shape -- another model's bicycle times, or a
distance matrix turned into walking times at a fixed speed -- and this model's
times overwrite it wherever they exist.

A model time "exists" for a pair when both zones are in the routed matrix and
the time is finite (reached within the cutoff). ``max_minutes`` narrows that
further: a routed time longer than it keeps the baseline instead, which is the
way to keep, say, trips that only route across an island connector at their
baseline value.

Everything is in minutes: the routed ``.npz`` holds seconds and is converted
on read, and a distance baseline is converted at its speed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_omx_dense(
    path: Path, matrix_name: str, mapping_name: str = "zone_number"
) -> tuple[np.ndarray, np.ndarray]:
    """(ids, float32 matrix) of one OMX matrix, read whole."""
    import openmatrix as omx

    with omx.open_file(str(path), "r") as f:
        if matrix_name not in f.list_matrices():
            raise ValueError(
                f"{path} has no matrix {matrix_name!r}; available: {f.list_matrices()}"
            )
        if mapping_name not in f.list_mappings():
            raise ValueError(
                f"{path} has no lookup {mapping_name!r}; available: {f.list_mappings()}"
            )
        ids = np.asarray(f.map_entries(mapping_name), dtype=np.int64)
        matrix = np.asarray(f[matrix_name][:], dtype=np.float32)
    if matrix.shape != (ids.size, ids.size):
        raise ValueError(
            f"{path}'s {matrix_name!r} matrix is {matrix.shape}, but its "
            f"{mapping_name!r} lookup has {ids.size} entries."
        )
    return ids, matrix


def read_model_minutes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ids, minutes) from a ``travel_times_<mode>.npz`` written by ``valma matrix``."""
    with np.load(path) as data:
        ids = data["ids"]
        minutes = data["seconds"].astype(np.float32) / np.float32(60.0)
    try:
        ids = ids.astype(np.int64)
    except ValueError as exc:
        raise ValueError(
            f"{path} has non-integer zone ids; they cannot be matched to an OMX lookup."
        ) from exc
    return ids, minutes


def overlay(
    base_ids: np.ndarray,
    base: np.ndarray,
    model_ids: np.ndarray,
    model: np.ndarray,
    max_minutes: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    ``base`` with ``model``'s values written over it wherever they apply.

    Returns the new matrix, in ``base_ids`` order and shape, and the number of
    cells replaced. ``base`` is not modified. Model zones missing from
    ``base_ids`` are ignored -- the baseline defines the zone system.
    """
    base_ids = np.asarray(base_ids)
    model_ids = np.asarray(model_ids)

    order = np.argsort(base_ids)
    pos = np.searchsorted(base_ids, model_ids, sorter=order)
    pos = np.clip(pos, 0, base_ids.size - 1)
    found = base_ids[order[pos]] == model_ids
    missing = int((~found).sum())
    if missing:
        logger.warning(
            "%d model zone(s) are not in the baseline's zone system; ignored "
            "(e.g. %s)",
            missing,
            model_ids[~found][:5].tolist(),
        )

    rows = order[pos[found]]
    sub_model = model[np.ix_(found, found)]
    use = np.isfinite(sub_model)
    if max_minutes is not None:
        use &= sub_model <= max_minutes

    result = np.array(base, dtype=np.float32, copy=True)
    block = result[np.ix_(rows, rows)]
    block[use] = sub_model[use]
    result[np.ix_(rows, rows)] = block
    return result, int(use.sum())
