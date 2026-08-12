"""Loading an OD demand matrix and aligning it to a set of centroids.

Demand between many zones is usually mostly zero, so this reads and returns
it as a SciPy sparse matrix throughout -- never as a dense (n_origins x
n_destinations) array. That distinction stops being cosmetic once zone counts
run into the thousands: a dense (1e4 x 1e4) float64 matrix is 800 MB on its
own, on top of whatever the network itself costs, and almost all of that
would be structural zero anyway.

Three source formats are supported: a long/triple-format CSV or similar
(:func:`read_demand_long` + :func:`demand_matrix`), a dense ``.npz``
(:func:`read_demand_npz`), and OMX -- the HDF5-based matrix format transport
models exchange OD matrices in (:func:`read_demand_omx`). :func:`clip_minimum`
applies a minimum-demand threshold uniformly across all three.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

logger = logging.getLogger(__name__)


def read_demand_long(
    path: Path,
    origin_column: str = "origin_id",
    destination_column: str = "destination_id",
    demand_column: str = "demand",
) -> pd.DataFrame:
    """
    Read an OD demand list: one row per (origin, destination, demand) triple.

    This "long" shape is the natural one for demand that is mostly zero,
    since it only stores what is nonzero -- unlike a dense matrix, whose size
    is fixed at n_origins * n_destinations whatever the demand looks like.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep=None, engine="python")
    missing = [
        c
        for c in (origin_column, destination_column, demand_column)
        if c not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path} has no column(s) {missing}; available: {list(frame.columns)}"
        )
    return frame[[origin_column, destination_column, demand_column]].rename(
        columns={
            origin_column: "origin_id",
            destination_column: "destination_id",
            demand_column: "demand",
        }
    )


def demand_matrix(long_demand: pd.DataFrame, ids: np.ndarray) -> csr_matrix:
    """
    Build a sparse ``(len(ids) x len(ids))`` demand matrix from a long-format list.

    ``ids`` is normally ``Centroids.ids``, so the result lines up positionally
    with those same centroids' ``node_index`` -- ready for
    :func:`valma_bike_and_walk.assignment.assign_traffic`. Rows referencing an
    id outside ``ids`` are dropped (and logged) rather than failing the whole
    run over one bad row, mirroring how ``Centroids.drop_unsnapped`` handles
    points the network can't reach. Repeated (origin, destination) rows are
    summed, which is convenient for demand assembled from several time
    periods or purposes.
    """
    index = pd.Index(ids)
    origin_pos = index.get_indexer(long_demand["origin_id"].to_numpy())
    dest_pos = index.get_indexer(long_demand["destination_id"].to_numpy())

    valid = (origin_pos >= 0) & (dest_pos >= 0)
    dropped = int((~valid).sum())
    if dropped:
        logger.warning(
            "Dropping %d demand row(s) referencing an id outside the %d given centroids",
            dropped,
            len(ids),
        )

    n = len(ids)
    matrix = coo_matrix(
        (
            long_demand["demand"].to_numpy(dtype=float)[valid],
            (origin_pos[valid], dest_pos[valid]),
        ),
        shape=(n, n),
    ).tocsr()
    matrix.sum_duplicates()
    logger.info("Built a %dx%d demand matrix, %d nonzero OD pair(s)", n, n, matrix.nnz)
    return matrix


def read_demand_npz(path: Path) -> tuple[np.ndarray, csr_matrix]:
    """
    Read a dense OD matrix saved with ``ids``/``demand`` arrays and sparsify it.

    For interop with demand already produced as a dense matrix (e.g. by a
    gravity model). Prefer :func:`read_demand_long` for anything assembled by
    hand or exported from a modelling tool -- it never needs the dense form.
    """
    path = Path(path)
    with np.load(path) as data:
        if "ids" not in data.files or "demand" not in data.files:
            raise ValueError(f"{path} must contain 'ids' and 'demand' arrays.")
        ids = data["ids"]
        dense = data["demand"]
    return ids, csr_matrix(dense.astype(np.float64))


#: Row-block size for OMX reads is chosen from this memory budget, the same
#: way matrix.py sizes its Dijkstra chunks -- see read_demand_omx.
_OMX_ROW_BUDGET_BYTES = 256 * 1024**2


def _omx_row_chunk(n_cols: int, budget_bytes: int = _OMX_ROW_BUDGET_BYTES) -> int:
    per_row = max(1, n_cols * 8)  # read as float64
    return max(1, min(n_cols, budget_bytes // per_row))


def read_demand_omx(
    path: Path,
    matrix_name: str,
    mapping_name: str = "zone_number",
    minimum_demand: float = 0.0,
) -> tuple[np.ndarray, csr_matrix]:
    """
    Read one matrix from an OMX (Open Matrix Format) file and sparsify it.

    OMX is the HDF5-based format transport models commonly exchange OD
    matrices in: a dense zones-by-zones array per travel mode (e.g. "bike"),
    plus one or more "lookup" arrays that give the row/column order meaning
    (here, ``zone_number`` -- a zone id per row/column position).

    The matrix is read a row-block at a time -- never all at once -- and
    ``minimum_demand`` is applied as each block comes in, so a matrix that
    is mostly (near-)zero, which is typical once there are thousands of
    zones, never needs its full dense size in memory at once. Requires the
    optional ``openmatrix`` package.
    """
    try:
        import openmatrix as omx
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading OMX demand needs the optional 'openmatrix' package: "
            "pip install openmatrix (or install this project's 'omx' extra)."
        ) from exc

    path = Path(path)
    with omx.open_file(str(path), "r") as f:
        if matrix_name not in f.list_matrices():
            raise ValueError(
                f"{path} has no matrix {matrix_name!r}; available: {f.list_matrices()}"
            )
        if mapping_name not in f.list_mappings():
            raise ValueError(
                f"{path} has no lookup {mapping_name!r}; available: {f.list_mappings()}"
            )

        ids = np.asarray(f.map_entries(mapping_name))
        node = f[matrix_name]
        n = int(node.shape[0])
        if node.shape[1] != n or n != len(ids):
            raise ValueError(
                f"{path}'s {matrix_name!r} matrix is {tuple(node.shape)}, but its "
                f"{mapping_name!r} lookup has {len(ids)} entries; they should match."
            )

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        values: list[np.ndarray] = []
        chunk = _omx_row_chunk(n)
        for start in range(0, n, chunk):
            block = np.asarray(node[start : start + chunk, :], dtype=np.float64)
            keep = (block > 0) & (block >= minimum_demand)
            r, c = np.nonzero(keep)
            rows.append(r + start)
            cols.append(c)
            values.append(block[r, c])

    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    logger.info(
        "Read %r from %s: %dx%d, %d nonzero OD pair(s) at or above minimum demand %g",
        matrix_name,
        path,
        n,
        n,
        matrix.nnz,
        minimum_demand,
    )
    return ids, matrix


def align_to_ids(
    ids: np.ndarray, matrix: csr_matrix, target_ids: np.ndarray
) -> csr_matrix:
    """
    Reindex a sparse OD matrix from its own id order onto ``target_ids``.

    A demand matrix's own ids (e.g. an OMX file's zone numbers, or a dense
    ``.npz``'s ``ids`` array) aren't guaranteed to be in the same order --
    or even cover the same set -- as the centroids being routed. This
    reorders (and drops, logging how many) rows/columns that don't match, so
    the result lines up with ``target_ids`` positionally, exactly like
    :func:`demand_matrix` already produces straight from a long-format file.
    """
    source_index = pd.Index(ids)
    target_pos = source_index.get_indexer(target_ids)
    found = target_pos >= 0
    missing = int((~found).sum())
    if missing:
        logger.warning(
            "%d of %d given centroid id(s) have no matching row in the demand "
            "matrix; treated as having zero demand.",
            missing,
            len(target_ids),
        )

    source_to_target = np.full(len(ids), -1, dtype=np.int64)
    source_to_target[target_pos[found]] = np.flatnonzero(found)

    coo = matrix.tocoo()
    new_row = source_to_target[coo.row]
    new_col = source_to_target[coo.col]
    keep = (new_row >= 0) & (new_col >= 0)

    n = len(target_ids)
    aligned = coo_matrix(
        (coo.data[keep], (new_row[keep], new_col[keep])), shape=(n, n)
    ).tocsr()
    aligned.sum_duplicates()
    return aligned


def clip_minimum(matrix: csr_matrix, minimum: float) -> csr_matrix:
    """
    Zero out (and drop) OD pairs with demand below ``minimum``.

    Assigning a shortest path costs roughly the same whether the OD pair
    carries 1000 units of demand or 0.01, so on a large zone system a lot of
    that cost can go to flows too small to matter. Screening them out first
    is standard practice with demand matrices this size, and trades a
    (normally tiny) amount of assigned volume for a cut in how many paths
    need building.
    """
    if minimum <= 0 or matrix.nnz == 0:
        return matrix
    below = matrix.data < minimum
    dropped = int(below.sum())
    if dropped:
        logger.info(
            "Clipping %d of %d OD pair(s) below minimum demand %g (%.3g total demand)",
            dropped,
            matrix.nnz,
            minimum,
            float(matrix.data[below].sum()),
        )
    matrix = matrix.copy()
    matrix.data[below] = 0
    matrix.eliminate_zeros()
    return matrix
