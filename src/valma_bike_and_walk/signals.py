"""Signalled junctions: which links pay the wait at a light.

OSM maps a signalled junction as many nodes -- a ``highway=traffic_signals``
at each stop line, a ``crossing=traffic_signals`` on each arm, one on each
carriageway of a divided road -- often four to eight of them, sometimes
dozens. Here they are grouped into junctions: signal nodes within
:data:`JUNCTION_RADIUS_M` of each other are one junction, and a link end
within that distance of one of its signal nodes is inside it.

Who pays, always in the direction of travel that does it:

- **A road entering the junction**, on the link that crosses its edge. Links
  inside it, and leaving it, pay nothing: once per junction, however many
  signal nodes the road passes.
- **A path stepping onto a signalled crossing**, on the link reaching the
  crossing's first signal node. A crossing over a divided road has a signal
  node on each carriageway; the links between them are the middle of the one
  crossing, and stepping off is free. A path going round the corner without
  crossing anything pays nothing: a right turn on the cycleway is free.
- **A path, or a service road, getting onto the junction's roads inside it**,
  on the link reaching the road. Without this, riding the cycleway into the
  junction and joining the road there would pass the light for free. A path
  reaching the road at a crossing node is crossing it, and pays as a crossing
  (signalled) or not at all (unsignalled, a zebra over a slip lane say).

Nothing more is paid by a link that comes from a signal node or one of the
junction's roads -- whoever is there has paid already. A cycleway crossing a
divided road over untagged nodes pays once, not once per carriageway, and
does not pay again for a signal node a few metres on.

An unsignalled crossing inside a junction is where crossing the road and
joining it meet at one node, and a link cannot see which a route does next.
:func:`split_crossing_entries` gives the paths there a node of their own,
joined to the road by a :data:`CROSSING_ENTRY_HIGHWAY` connector that pays the
junction on the way onto the road. Crossing stays free, joining does not.

Every one of these is the same wait, so no way in is cheaper than another:
stepping onto a crossing and turning onto the road there costs what the road
approach does.

A mid-block pedestrian light -- signalled crossings with no road light and no
side road -- is a junction of its own kind, :data:`MIDBLOCK`, with a smaller
wait. Its inside is its signal nodes alone, so a sidewalk running past it is
free; a road passing it and a path crossing it each pay once.

What a code costs in seconds lives in :mod:`valma_bike_and_walk.links`, so it
can be tuned without re-extracting. Only the codes are worked out here, once,
at extract time, because they need the whole network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

#: Signal nodes this close together are one junction, and a link end this close
#: to one of a junction's signal nodes is inside the junction. Wide enough for
#: the stop lines and crossings of a large urban junction; narrow enough that
#: two junctions a block apart stay two.
JUNCTION_RADIUS_M = 30.0

#: ``highway`` of the connector joining a split crossing's paths to its road.
CROSSING_ENTRY_HIGHWAY = "crossing_entry"

#: Codes stored per link and direction: what kind of light the link pays for.
#: 2 and 4 were used by an earlier scheme and are read as JUNCTION.
NONE, JUNCTION, MIDBLOCK = 0, 1, 3

#: The roads a junction is made of: ``unclassified``/``residential`` and up,
#: their ``_link`` slip roads included. Service roads and living streets are
#: not -- a driveway or parking aisle inside the radius is not the junction, and
#: a sidewalk passing its end should not pay.
JUNCTION_ROADS = frozenset(
    {
        "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link",
        "unclassified", "residential",
    }
)


@dataclass(frozen=True)
class _Layout:
    """Signal nodes grouped into junctions, and where each link end sits."""

    node_ids: np.ndarray  # sorted
    ui: np.ndarray  # position of each link's u in node_ids
    vi: np.ndarray
    ju: np.ndarray  # junction each link's u is inside, -1 outside every one
    jv: np.ndarray
    path: np.ndarray
    junction_road: np.ndarray
    code: np.ndarray  # per junction: JUNCTION or MIDBLOCK
    is_signal: np.ndarray  # per node
    is_crossing: np.ndarray  # per node: signalled or not
    on_junction_road: np.ndarray  # per node: one of its junction's roads runs here


def _layout(
    u, v, highway, path, node_ids, node_x, node_y,
    signal_ids, light_ids, crossing_ids, radius_m,
) -> _Layout | None:
    """The junctions of a link layer, or None when it has no signal nodes."""
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

    def flags(ids: np.ndarray | None) -> np.ndarray:
        is_one = np.zeros(len(node_ids), dtype=bool)
        if ids is not None and len(ids):
            at = index(np.unique(np.asarray(ids, dtype=np.int64)))
            is_one[at[at >= 0]] = True
        return is_one

    is_signal = flags(signal_ids)
    signals = np.flatnonzero(is_signal)  # signal nodes that are link ends
    if signals.size == 0:
        return None

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
    inside = np.where(
        np.isfinite(distance), label[np.minimum(nearest, signals.size - 1)], -1
    )

    ui = index(np.asarray(u, dtype=np.int64))
    vi = index(np.asarray(v, dtype=np.int64))
    path = np.asarray(path, dtype=bool)
    road = ~path
    junction_road = np.isin(np.asarray(highway, dtype=object), list(JUNCTION_ROADS))

    # Mid-block: no road light, and the road runs straight through every
    # signal node -- no side road.
    degree = np.zeros(len(node_ids), dtype=np.int64)
    for i in (ui, vi):
        np.add.at(degree, i[road], 1)
    has_light = (
        np.bincount(label, weights=flags(light_ids)[signals], minlength=n_junctions) > 0
    )
    max_degree = np.zeros(n_junctions, dtype=np.int64)
    np.maximum.at(max_degree, label, degree[signals])
    midblock = ~has_light & (max_degree <= 2)

    # A mid-block light's inside is its signal nodes: passing beside it is free.
    inside = np.where(midblock[np.maximum(inside, 0)] & ~is_signal, -1, inside)
    code = np.where(midblock, MIDBLOCK, JUNCTION).astype(np.int8)
    ju, jv = inside[ui], inside[vi]

    # The junction's roads: nodes inside it where one of its roads runs.
    on_junction_road = np.zeros(len(node_ids), dtype=bool)
    for i, j in ((ui, ju), (vi, jv)):
        on_junction_road[i[junction_road & (j >= 0)]] = True
    return _Layout(
        node_ids=node_ids, ui=ui, vi=vi, ju=ju, jv=jv, path=path,
        junction_road=junction_road, code=code, is_signal=is_signal,
        is_crossing=is_signal | flags(crossing_ids),
        on_junction_road=on_junction_road,
    )


@dataclass(frozen=True)
class CrossingSplit:
    """Path ends moved off crossing nodes, and the connectors that rejoin them.

    ``u``/``v`` are the links' ends with every path end at a split node moved
    to that node's path-side twin. Connector ``k`` runs from
    ``connector_from[k]`` (the twin, path side) to ``connector_to[k]`` (the
    crossing node itself, on the road), at ``connector_x[k]``,
    ``connector_y[k]``.
    """

    u: np.ndarray
    v: np.ndarray
    connector_from: np.ndarray
    connector_to: np.ndarray
    connector_x: np.ndarray
    connector_y: np.ndarray

    @property
    def count(self) -> int:
        return int(self.connector_from.shape[0])


def split_crossing_entries(
    u: np.ndarray,
    v: np.ndarray,
    highway: np.ndarray,
    path: np.ndarray,
    node_ids: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    signal_ids: np.ndarray,
    light_ids: np.ndarray,
    crossing_ids: np.ndarray,
    radius_m: float = JUNCTION_RADIUS_M,
) -> CrossingSplit:
    """
    Give the paths at an unsignalled crossing inside a junction their own node.

    A path reaching one of a junction's roads at a crossing is crossing it, and
    a crossing without lights is free -- but the same node is also where a
    cyclist can step off the path onto that road and ride through the junction
    without ever reaching a stop line. A link is charged on its own, with no
    idea where the route goes next, so the two cannot be told apart at the
    node. Here they are pulled apart: the paths at such a node move to a twin
    of it, joined to the road by a connector. Path to path across the road
    stays free; path onto the road goes through the connector, which pays the
    junction (:func:`signal_codes` charges it as a link joining the junction's
    roads); road onto path is free.
    """
    lay = _layout(
        u, v, highway, path, node_ids, node_x, node_y,
        signal_ids, light_ids, crossing_ids, radius_m,
    )
    u = np.asarray(u, dtype=np.int64).copy()
    v = np.asarray(v, dtype=np.int64).copy()
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    if lay is None:
        return CrossingSplit(u, v, empty_i, empty_i, empty_f, empty_f)

    touched_by_path = np.zeros(len(lay.node_ids), dtype=bool)
    touched_by_path[lay.ui[lay.path]] = True
    touched_by_path[lay.vi[lay.path]] = True
    split = (
        lay.is_crossing & ~lay.is_signal & lay.on_junction_road & touched_by_path
    )
    at = np.flatnonzero(split)
    if at.size == 0:
        return CrossingSplit(u, v, empty_i, empty_i, empty_f, empty_f)

    # New ids below every existing one: negative, as for nodes drawn in QGIS.
    base = min(int(lay.node_ids.min()), 0) - 1
    twin_of = np.full(len(lay.node_ids), 0, dtype=np.int64)
    twin_of[at] = base - np.arange(at.size, dtype=np.int64)
    move_u = lay.path & split[lay.ui]
    move_v = lay.path & split[lay.vi]
    u[move_u] = twin_of[lay.ui[move_u]]
    v[move_v] = twin_of[lay.vi[move_v]]

    order = np.argsort(np.asarray(node_ids, dtype=np.int64))
    xs = np.asarray(node_x, dtype=float)[order]
    ys = np.asarray(node_y, dtype=float)[order]
    return CrossingSplit(
        u=u,
        v=v,
        connector_from=twin_of[at],
        connector_to=lay.node_ids[at],
        connector_x=xs[at],
        connector_y=ys[at],
    )


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
    crossing_ids: np.ndarray | None = None,
    radius_m: float = JUNCTION_RADIUS_M,
) -> dict[str, np.ndarray]:
    """
    The wait codes of every link, from the layer and its signal nodes.

    ``u``/``v`` are each link's end node ids; ``highway`` its class; ``path``
    whether it is a cycleway/footway/path kind of link. ``node_ids`` with
    ``node_x``/``node_y`` place every link end in metres. ``signal_ids`` are all
    signal nodes, ``light_ids`` the subset that are road lights
    (``highway=traffic_signals``) rather than signalled crossings.
    ``crossing_ids`` are unsignalled crossing nodes.

    Returns int8 code arrays, each charged in one direction of travel:
    ``signal_entry`` / ``signal_entry_reverse`` for getting onto a junction's
    roads along / against the digitised direction, ``signal_cross_end`` /
    ``signal_cross_start`` for a path stepping onto a signalled crossing at
    that end -- the end it travels towards.
    """
    n = u.shape[0]
    out = {
        "signal_entry": np.zeros(n, dtype=np.int8),
        "signal_entry_reverse": np.zeros(n, dtype=np.int8),
        "signal_cross_start": np.zeros(n, dtype=np.int8),
        "signal_cross_end": np.zeros(n, dtype=np.int8),
    }
    lay = _layout(
        u, v, highway, path, node_ids, node_x, node_y,
        signal_ids, light_ids, crossing_ids, radius_m,
    )
    if lay is None:
        return out
    ui, vi, ju, jv = lay.ui, lay.vi, lay.ju, lay.jv
    path, junction_road, code = lay.path, lay.junction_road, lay.code
    is_signal, is_crossing, on_junction_road = (
        lay.is_signal, lay.is_crossing, lay.on_junction_road
    )
    # Where whoever stands there has paid the junction already: on a signalled
    # crossing, or on one of its roads. Not at a zebra over a slip lane, which
    # is free to cross, or it would be a free way in.
    paid = is_signal | (on_junction_road & ~is_crossing)

    def entering(arrive: np.ndarray, junction: np.ndarray, leave: np.ndarray,
                 other: np.ndarray) -> np.ndarray:
        """Getting onto the junction's roads by travelling to ``arrive``."""
        # A junction road from outside: the stop line.
        enters = junction_road & (junction >= 0) & (junction != other)
        # Anything else reaching one of its roads inside it -- off a path, or
        # along a service road -- but a path reaching it at a crossing, and
        # not from where the junction is paid already.
        joins = (
            ~junction_road
            & (junction >= 0)
            & on_junction_road[arrive]
            & ~(path & is_crossing[arrive])
            & ~(paid[leave] & (other == junction))
        )
        return np.where(enters | joins, code[np.maximum(junction, 0)], NONE).astype(np.int8)

    out["signal_entry"] = entering(vi, jv, ui, ju)          # u -> v
    out["signal_entry_reverse"] = entering(ui, ju, vi, jv)  # v -> u

    def stepping_on(arrive: np.ndarray, junction: np.ndarray, leave: np.ndarray,
                    other: np.ndarray) -> np.ndarray:
        """A path reaching a signalled crossing by travelling to ``arrive``."""
        # From a signal node of the same junction (already on the crossing) or
        # one of its roads (paid getting onto them): nothing more to pay.
        on = path & is_signal[arrive] & ~(paid[leave] & (other == junction))
        return np.where(on, code[np.maximum(junction, 0)], NONE).astype(np.int8)

    out["signal_cross_end"] = stepping_on(vi, jv, ui, ju)    # u -> v
    out["signal_cross_start"] = stepping_on(ui, ju, vi, jv)  # v -> u
    return out
