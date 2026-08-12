"""Tests for reading and aligning an OD demand matrix."""

import numpy as np
import pandas as pd
import pytest

from scipy.sparse import csr_matrix

from valma_bike_and_walk.demand import (
    align_to_ids,
    clip_minimum,
    demand_matrix,
    read_demand_long,
    read_demand_npz,
    read_demand_omx,
)


def test_read_demand_long_renames_columns(tmp_path):
    path = tmp_path / "od.csv"
    path.write_text("from,to,trips\nA,B,3\nB,C,4\n")

    frame = read_demand_long(
        path, origin_column="from", destination_column="to", demand_column="trips"
    )

    assert list(frame.columns) == ["origin_id", "destination_id", "demand"]
    assert frame["demand"].tolist() == [3, 4]


def test_read_demand_long_missing_column_raises(tmp_path):
    path = tmp_path / "od.csv"
    path.write_text("origin_id,destination_id\nA,B\n")

    with pytest.raises(ValueError, match="demand"):
        read_demand_long(path)


def test_demand_matrix_aligns_to_given_ids():
    long_demand = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["B", "C"],
            "demand": [3.0, 4.0],
        }
    )
    ids = np.array(["A", "B", "C"])

    matrix = demand_matrix(long_demand, ids)

    assert matrix.shape == (3, 3)
    dense = matrix.toarray()
    assert dense[0, 1] == pytest.approx(3.0)
    assert dense[1, 2] == pytest.approx(4.0)
    assert matrix.nnz == 2


def test_demand_matrix_sums_duplicate_rows():
    long_demand = pd.DataFrame(
        {
            "origin_id": ["A", "A"],
            "destination_id": ["B", "B"],
            "demand": [3.0, 4.0],
        }
    )
    ids = np.array(["A", "B"])

    matrix = demand_matrix(long_demand, ids)

    assert matrix.toarray()[0, 1] == pytest.approx(7.0)
    assert matrix.nnz == 1


def test_demand_matrix_drops_unknown_ids(caplog):
    long_demand = pd.DataFrame(
        {
            "origin_id": ["A", "Z"],
            "destination_id": ["B", "B"],
            "demand": [3.0, 4.0],
        }
    )
    ids = np.array(["A", "B"])

    with caplog.at_level("WARNING"):
        matrix = demand_matrix(long_demand, ids)

    assert matrix.nnz == 1
    assert matrix.toarray()[0, 1] == pytest.approx(3.0)
    assert any("Dropping" in message for message in caplog.messages)


def test_read_demand_npz_round_trips(tmp_path):
    path = tmp_path / "od.npz"
    ids = np.array([1, 2, 3])
    dense = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
    np.savez_compressed(path, ids=ids, demand=dense)

    read_ids, matrix = read_demand_npz(path)

    np.testing.assert_array_equal(read_ids, ids)
    np.testing.assert_allclose(matrix.toarray(), dense)


def test_read_demand_npz_requires_both_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, ids=np.array([1, 2]))

    with pytest.raises(ValueError, match="ids.*demand"):
        read_demand_npz(path)


def _write_omx(path, ids, dense, mapping_name="zone_number", matrix_name="bike"):
    omx = pytest.importorskip("openmatrix")
    with omx.open_file(str(path), "w") as f:
        f.create_mapping(mapping_name, list(ids))
        f.create_matrix(matrix_name, obj=dense.astype("float32"))


def test_read_demand_omx_round_trips(tmp_path):
    path = tmp_path / "od.omx"
    ids = [10, 20, 30]
    dense = np.array([[0.0, 1.5, 0.0], [2.0, 0.0, 0.4], [0.0, 0.0, 0.0]])
    _write_omx(path, ids, dense)

    read_ids, matrix = read_demand_omx(path, matrix_name="bike")

    np.testing.assert_array_equal(read_ids, np.array(ids))
    np.testing.assert_allclose(matrix.toarray(), dense, atol=1e-6)


def test_read_demand_omx_applies_minimum_at_read_time(tmp_path):
    path = tmp_path / "od.omx"
    ids = [1, 2, 3]
    dense = np.array([[0.0, 0.4, 0.0], [2.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    _write_omx(path, ids, dense)

    _ids, matrix = read_demand_omx(path, matrix_name="bike", minimum_demand=1.0)

    assert matrix.nnz == 2
    np.testing.assert_allclose(sorted(matrix.data), [2.0, 5.0])


def test_read_demand_omx_unknown_matrix_raises(tmp_path):
    path = tmp_path / "od.omx"
    _write_omx(path, [1, 2], np.zeros((2, 2)))

    with pytest.raises(ValueError, match="no matrix"):
        read_demand_omx(path, matrix_name="car")


def test_read_demand_omx_unknown_mapping_raises(tmp_path):
    path = tmp_path / "od.omx"
    _write_omx(path, [1, 2], np.zeros((2, 2)))

    with pytest.raises(ValueError, match="no lookup"):
        read_demand_omx(path, matrix_name="bike", mapping_name="taz_id")


def test_align_to_ids_reorders_and_drops_unmatched(caplog):
    # Source order is [A, B, C], with demand A->C=5 and B->A=2. Target order
    # is [C, A, D]: D isn't a source id at all, and B (which only appears as
    # a B->A trip here) isn't in the target set either.
    matrix = csr_matrix(np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    ids = np.array(["A", "B", "C"])
    target = np.array(["C", "A", "D"])

    with caplog.at_level("WARNING"):
        aligned = align_to_ids(ids, matrix, target)

    assert aligned.shape == (3, 3)
    dense = aligned.toarray()
    # A->C survives, reindexed to (target index of A, target index of C) = (1, 0).
    assert dense[1, 0] == pytest.approx(5.0)
    # B->A is dropped entirely, since B has no row/column in the target ids.
    assert aligned.nnz == 1
    assert any("no matching row" in message for message in caplog.messages)


def test_align_to_ids_preserves_values_for_matched_pairs():
    matrix = csr_matrix(np.array([[0.0, 5.0], [0.0, 0.0]]))
    ids = np.array(["A", "B"])
    target = np.array(["B", "A"])  # swapped order

    aligned = align_to_ids(ids, matrix, target)

    # source A->B (5.0) is target index 1 -> index 0.
    assert aligned.toarray()[1, 0] == pytest.approx(5.0)
    assert aligned.nnz == 1


def test_clip_minimum_drops_small_entries():
    matrix = csr_matrix(np.array([[0.0, 0.5, 2.0], [0.0, 0.0, 0.1]]))

    clipped = clip_minimum(matrix, 1.0)

    assert clipped.nnz == 1
    np.testing.assert_allclose(clipped.data, [2.0])


def test_clip_minimum_noop_when_minimum_not_positive():
    matrix = csr_matrix(np.array([[0.0, 0.5], [0.2, 0.0]]))
    clipped = clip_minimum(matrix, 0.0)
    assert clipped.nnz == matrix.nnz
