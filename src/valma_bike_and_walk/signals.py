"""Signalled junctions: which lights a route waits at, and for how long.

OSM maps a signalled junction as many nodes -- a ``highway=traffic_signals``
at each stop line, a ``crossing=traffic_signals`` on each arm -- often four to
eight of them, sometimes dozens. Charging a wait at every one of those nodes
charged a route several waits for one junction. Here the nodes are grouped
into junctions first, and the wait is charged per junction:

- **On the road**, once, on the link that *enters* the junction, in the
  direction that enters it. A link end within :data:`JUNCTION_RADIUS_M` of one
  of the junction's signal nodes is inside it, so a road through the junction
  -- stop line, centre, the crossing on the far arm -- enters it only once.
- **On a path**, at every signalled crossing actually crossed, half on each
  link touching the crossing node, as before. Crossing two arms really is two
  waits.

How long the wait is depends on how much green the movement gets, which is
read from the road classes (see :data:`APPROACH_CODES`):

- an approach on the junction's main road, or a path crossing a *side* road
  (it gets green with the main road), has the main share of the cycle;
- an approach from a smaller road, or a path crossing the *main* road, has the
  minor share;
- a mid-block pedestrian light -- signalled crossings with no road light and
  no side road -- barely stops the road (it waits for a push), and makes the
  one crossing it wait for its turn.

What a code costs in seconds lives in :mod:`valma_bike_and_walk.links`, so it
can be tuned without re-extracting. Only the codes are worked out here, once,
at extract time, because they need the whole network.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

#: Signal nodes this close together are one junction, and a link end this close
#: to one of a junction's signal nodes is inside the junction. Wide enough for
#: the stop lines and crossings of a large urban junction; narrow enough that
#: two junctions a block apart stay two.
JUNCTION_RADIUS_M = 30.0

#: Codes stored per link: what kind of wait a traversal meets. EQUAL is a
#: junction of equals -- its top road class arrives from three or more
#: directions, a crossroads of two secondaries say -- where the green is shared
#: rather than given to one main road.
NONE, MAIN, MINOR, MIDBLOCK, EQUAL = 0, 1, 2, 3, 4
APPROACH_CODES = {
    NONE: "none", MAIN: "main", MINOR: "minor", MIDBLOCK: "mid-block", EQUAL: "equal"
}

#: Arms arriving within this many degrees of each other are one direction: the
#: two carriageways of a divided road are one arm, not two.
ARM_SEPARATION_DEG = 45.0

#: Road classes by priority, for telling a junction's main road from its side
#: roads. Links count as their road (``primary_link`` as ``primary``).
ROAD_RANK = {
    "trunk": 6, "trunk_link": 6,
    "primary": 5, "primary_link": 5,
    "secondary": 4, "secondary_link": 4,
    "tertiary": 3, "tertiary_link": 3,
    "unclassified": 2, "residential": 2, "living_street": 1, "service": 1,
}


def signal_codes(
    u: np.ndarray,
    v: np.ndarray,
    highway: np.ndarray,
    path: np.ndarray,
    node_ids: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    signal_ids: np.ndarray,
    light_ids: np.ndarray,
    radius_m: float = JUNCTION_RADIUS_M,
) -> dict[str, np.ndarray]:
    """
    The wait codes of every link, from the layer and its signal nodes.

    ``u``/``v`` are each link's end node ids; ``highway`` its class; ``path``
    whether it is a cycleway/footway/path kind of link. ``node_ids`` with
    ``node_x``/``node_y`` place every link end in metres. ``signal_ids`` are all
    signal nodes, ``light_ids`` the subset that are road lights
    (``highway=traffic_signals``) rather than signalled crossings.

    Returns int8 arrays: ``signal_entry`` / ``signal_entry_reverse`` for road
    links entering a junction along / against their digitised direction, and
    ``signal_cross_start`` / ``signal_cross_end`` for path links with a signal
    node at that end.
    """
    n = u.shape[0]
    out = {
        "signal_entry": np.zeros(n, dtype=np.int8),
        "signal_entry_reverse": np.zeros(n, dtype=np.int8),
        "signal_cross_start": np.zeros(n, dtype=np.int8),
        "signal_cross_end": np.zeros(n, dtype=np.int8),
    }
    node_ids = np.asarray(node_ids, dtype=np.int64)
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    xy = np.column_stack([np.asarray(node_x)[order], np.asarray(node_y)[order]])

    def index(ids: np.ndarray) -> np.ndarray:
        """Position of each id in node_ids, -1 where it is not a link end."""
        if len(node_ids) == 0:
            return np.full(len(ids), -1, dtype=np.int64)
        at = np.minimum(np.searchsorted(node_ids, ids), len(node_ids) - 1)
        return np.where(node_ids[at] == ids, at, -1)

    signals = index(np.unique(np.asarray(signal_ids, dtype=np.int64)))
    signals = signals[signals >= 0]  # signal nodes that are link ends
    if signals.size == 0:
        return out

    # Signal nodes into junctions: connected within the radius.
    tree = cKDTree(xy[signals])
    pairs = tree.query_pairs(radius_m, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(signals.size,) * 2
    )
    _, label = connected_components(graph, directed=False)
    n_junctions = int(label.max()) + 1

    # Every link end inside a junction: its nearest signal node within reach.
    distance, nearest = tree.query(xy, distance_upper_bound=radius_m)
    junction_of_node = np.where(np.isfinite(distance), label[np.minimum(nearest, signals.size - 1)], -1)

    ui, vi = index(np.asarray(u, dtype=np.int64)), index(np.asarray(v, dtype=np.int64))
    ju, jv = junction_of_node[ui], junction_of_node[vi]
    rank = np.array([ROAD_RANK.get(str(h), -1) for h in highway], dtype=np.int64)
    road = rank >= 0

    # The junction's main road: the highest class touching it.
    top = np.full(n_junctions, -1, dtype=np.int64)
    for j in (ju, jv):
        inside = road & (j >= 0)
        np.maximum.at(top, j[inside], rank[inside])

    # Road links meeting each node, and the best class among them.
    degree = np.zeros(len(node_ids), dtype=np.int64)
    best = np.full(len(node_ids), -1, dtype=np.int64)
    for i in (ui, vi):
        np.add.at(degree, i[road], 1)
        np.maximum.at(best, i[road], rank[road])

    # Mid-block: no road light, and the road runs straight through every node.
    is_light = np.zeros(len(node_ids), dtype=bool)
    lights = index(np.unique(np.asarray(light_ids, dtype=np.int64)))
    is_light[lights[lights >= 0]] = True
    has_light = np.bincount(label, weights=is_light[signals], minlength=n_junctions) > 0
    max_degree = np.zeros(n_junctions, dtype=np.int64)
    np.maximum.at(max_degree, label, degree[signals])
    midblock = ~has_light & (max_degree <= 2)
    equal = _equal_junctions(
        label, xy[signals], ju, jv, ui, vi, xy, road & (rank >= 0), rank, top
    )

    def approach(junction: np.ndarray, other: np.ndarray) -> np.ndarray:
        enters = road & (junction >= 0) & (junction != other)
        j = np.maximum(junction, 0)
        code = np.where(rank >= top[j], np.where(equal[j], EQUAL, MAIN), MINOR)
        code = np.where(midblock[j], MIDBLOCK, code)
        return np.where(enters, code, NONE).astype(np.int8)

    out["signal_entry"] = approach(jv, ju)          # u -> v enters v's junction
    out["signal_entry_reverse"] = approach(ju, jv)  # v -> u enters u's junction

    is_signal = np.zeros(len(node_ids), dtype=bool)
    is_signal[signals] = True

    def crossing(i: np.ndarray, junction: np.ndarray) -> np.ndarray:
        at_signal = path & (i >= 0) & is_signal[np.maximum(i, 0)]
        j = np.maximum(junction, 0)
        # The road being crossed is the one through this node; a crossing node
        # off the road is taken to cross the junction's main road.
        crossed = np.where(best[np.maximum(i, 0)] >= 0, best[np.maximum(i, 0)], top[j])
        code = np.where(crossed >= top[j], np.where(equal[j], EQUAL, MINOR), MAIN)
        code = np.where(midblock[j], MIDBLOCK, code)
        return np.where(at_signal, code, NONE).astype(np.int8)

    out["signal_cross_start"] = crossing(ui, ju)
    out["signal_cross_end"] = crossing(vi, jv)
    return out


def _equal_junctions(
    label: np.ndarray,
    signal_xy: np.ndarray,
    ju: np.ndarray,
    jv: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    xy: np.ndarray,
    road: np.ndarray,
    rank: np.ndarray,
    top: np.ndarray,
) -> np.ndarray:
    """
    Junctions whose top road class arrives from three or more directions.

    An arm is a top-class road link with one end inside the junction and one
    outside, placed by the bearing of its outside end from the junction's
    centre. Bearings within :data:`ARM_SEPARATION_DEG` are one direction.
    """
    n = int(label.max()) + 1
    centre = np.zeros((n, 2))
    np.add.at(centre, label, signal_xy)
    centre /= np.bincount(label, minlength=n)[:, None]

    junctions, bearings = [], []
    for inside, outside, j_in, j_out in ((vi, ui, jv, ju), (ui, vi, ju, jv)):
        arm = road & (j_in >= 0) & (j_in != j_out)
        arm &= rank >= top[np.maximum(j_in, 0)]
        j = j_in[arm]
        dx, dy = (xy[outside[arm]] - centre[j]).T
        junctions.append(j)
        bearings.append(np.degrees(np.arctan2(dy, dx)) % 360.0)
    junction = np.concatenate(junctions)
    bearing = np.concatenate(bearings)

    equal = np.zeros(n, dtype=bool)
    order = np.lexsort((bearing, junction))
    junction, bearing = junction[order], bearing[order]
    starts = np.flatnonzero(np.r_[True, junction[1:] != junction[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], junction.size]):
        angles = bearing[start:stop]
        if angles.size < 3:
            continue
        gaps = np.diff(np.r_[angles, angles[0] + 360.0])
        # Directions = gaps wide enough to separate two arms (wrapping round).
        equal[junction[start]] = int((gaps > ARM_SEPARATION_DEG).sum()) >= 3
    return equal
