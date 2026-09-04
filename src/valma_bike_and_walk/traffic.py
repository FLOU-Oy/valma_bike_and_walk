"""Car traffic volumes from another network, conflated onto the link layer.

A traffic model's assignment result is a set of lines with a volume on each, and
those lines are not our lines: they come from a coarser network, digitised
separately, and they agree with OSM only to within a few metres. This module is
the map-matching step that carries the volume across -- so that a link the
pipeline routes bicycles over knows how many cars share it.

Three things make it more than a nearest-neighbour join.

**Only a road that carries cars may be given a volume.** A cycleway beside a main
road is often closer to the model's centreline than the carriageway itself is, so
proximity alone would load the traffic onto exactly the link it does not belong
on. Candidates are drawn only from :data:`CAR_HIGHWAYS`; the cycle track next to
a road is never one, whatever the distance.

**A model link need not exist here at all.** Motorways are filtered out of the
bicycle network, so a motorway's volume has nothing legitimate to land on -- and
a service road twenty metres away is a very tempting wrong answer. The defence is
coverage: a source link is matched *as a whole* or not at all. Its length is
sampled every :data:`DEFAULT_SPACING_M`, and unless
:attr:`MatchSettings.min_coverage` of those samples find a road, every claim it
made is thrown away. A motorway that brushes a service road for a hundred metres
of its kilometre fails that test, and the service road keeps its own volume.

**A divided road is two links here and one there.** The two carriageways are
digitised in opposite directions, which is what lets them be recognised: a sample
claims its best road, and then the best road running the *other* way, if both are
one-way. When a source is claimed from both directions the volume it carries is
the two carriageways' total, so each gets half -- otherwise a cyclist on one side
of the central reservation is charged for the traffic on the other. The check
that this is right is conservation: matched vehicle-kilometres come to 97.5% of
the source layer's own on the Espoo test area, against 117% without the split.

Columns added
-------------
``car_volume``            vehicles a day sharing this link, empty where nothing
                          matched -- which means either the link carries no cars
                          or the model has no link here, and ``highway`` says
                          which.
``car_volume_offset_m``   how far the model's line sat from ours, averaged over
                          the samples that matched. The one number to style the
                          layer by when checking a match by eye.
``car_volume_override``   empty for you to fill in, the same invitation
                          ``speed_override_kmh`` extends.

Where it goes in the pipeline
-----------------------------
After ``valma extract`` (and ``valma dem``), before the graph is built::

    valma extract --pbf finland.osm.pbf --mode bike --bbox ...
    valma traffic --links output/bike_links.gpkg --volumes data/link_volumes.gpkg
    valma matrix  --links output/bike_links.gpkg --mode bike ...

It is a link-layer stage rather than a graph one, for the same reason elevation
is: the columns then survive a QGIS edit, are re-read on every build, and can be
corrected by hand where the match went wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.links import ONEWAY_VALUES, geometry_lengths
from valma_bike_and_walk.progress import Progress

logger = logging.getLogger(__name__)

#: Way types that carry general motor traffic, and so may be given a volume.
#: Everything else in the bicycle network -- cycleway, footway, path, steps,
#: track, pedestrian, busway -- is excluded outright, which is what stops a
#: cycle track absorbing the volume of the road it runs beside.
CAR_HIGHWAYS = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "road",
    }
)

#: How far apart the source links are sampled, in metres. Half a short urban
#: link, so even a 40 m stub gets two independent votes, and fine enough that
#: coverage is a meaningful fraction rather than a coin toss.
DEFAULT_SPACING_M = 20.0

#: How far a sample may be from a road and still call it a match. The genuine
#: matches on the Espoo test area sit at a median of 0.9 m and a third quartile
#: of 1.7 m; the wrong ones -- a motorway reaching for the service road beside
#: it -- start at about 5 m. 12 m leaves room for a differently digitised
#: network without opening the door to the road next door.
DEFAULT_MAX_DISTANCE_M = 12.0

#: How far the two lines may differ in bearing. Undirected: which way each
#: network digitised the road is not a fact about the road.
DEFAULT_MAX_ANGLE_DEG = 45.0

#: The share of a source link's samples that must find a road before any of its
#: claims count. Coverage is sharply bimodal -- on the test area 75% of sources
#: score exactly 1.0 and 19% exactly 0.0 -- so the threshold has a wide valley to
#: sit in, and "at least half the link has to be there" is the rule that is
#: easiest to say out loud.
DEFAULT_MIN_COVERAGE = 0.5

#: Distance at which a sample's vote is worth 1/e of a coincident one. Closer
#: evidence outweighs more evidence, which is what settles a link two sources
#: both reach for.
DISTANCE_DECAY_M = 10.0

#: Each direction has to account for this much of a source's matched weight
#: before the source is read as a divided road. Keeps a few metres of one-way
#: slip road at a junction from halving a whole link's volume.
DIVIDED_MIN_SHARE = 0.25

#: Samples per chunk. The candidate query and its geometry work are proportional
#: to this, so it sets peak memory; a country's 13 million samples then cost a
#: predictable amount rather than all of RAM.
SAMPLES_PER_CHUNK = 1_000_000

VOLUME_COLUMN = "car_volume"
OFFSET_COLUMN = "car_volume_offset_m"
OVERRIDE_COLUMN = "car_volume_override"

#: Column read from the source layer unless another is named.
DEFAULT_VOLUME_COLUMN = "volume"


@dataclass(frozen=True)
class MatchSettings:
    """The knobs on the match. The defaults are the ones the module docstring argues for."""

    max_distance_m: float = DEFAULT_MAX_DISTANCE_M
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG
    min_coverage: float = DEFAULT_MIN_COVERAGE
    spacing_m: float = DEFAULT_SPACING_M
    #: Split a divided road's volume between its two carriageways.
    split_divided: bool = True

    def __post_init__(self) -> None:
        if self.max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive.")
        if not 0 < self.max_angle_deg < 90:
            raise ValueError("max_angle_deg must be between 0 and 90 degrees.")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be a fraction between 0 and 1.")
        if self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive.")


@dataclass(frozen=True)
class VolumeMatch:
    """
    What matched what.

    ``link_row`` and ``source_row`` are positional indices into the link layer
    and the volume layer as they were passed in -- not ``link_id``, which the
    caller may not have assigned yet. One row per matched link; a link appears at
    most once, carrying the volume of whichever source claimed most of it.
    """

    link_row: np.ndarray
    source_row: np.ndarray
    volume: np.ndarray
    offset_m: np.ndarray
    #: Whether the source a link took its volume from was read as divided, and so
    #: whether ``volume`` is half of what that source carries.
    divided: np.ndarray
    #: Per source row: the share of its samples that found a road at all, and
    #: ``nan`` for a source outside the link layer's extent, which was never
    #: asked the question.
    source_coverage: np.ndarray
    #: Per source row: whether it passed the coverage test and was used.
    source_used: np.ndarray
    #: Per source row: whether it lies within reach of the link layer at all. A
    #: national volume layer against a city's links is the normal case, and the
    #: rest of the country is not a failed match.
    source_in_extent: np.ndarray

    @property
    def n_links(self) -> int:
        return int(self.link_row.shape[0])


# --------------------------------------------------------------------------
# Reading the source layer
# --------------------------------------------------------------------------


def read_volumes(
    path: Path | str,
    layer: str | None = None,
    column: str = DEFAULT_VOLUME_COLUMN,
) -> gpd.GeoDataFrame:
    """
    Read a volume layer and reduce it to what the match needs: lines and a number.

    Multi-part geometries are exploded, because a sample walks one line at a
    time, and rows with no volume or no geometry are dropped rather than carried
    along as noise.
    """
    path = Path(path)
    volumes = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if column not in volumes.columns:
        raise ValueError(
            f"{path} has no {column!r} column; it has "
            f"{[c for c in volumes.columns if c != 'geometry']}. "
            "Name the right one with --volume-column."
        )
    if volumes.crs is None:
        raise ValueError(f"{path} has no CRS, so its lines cannot be placed.")

    volumes = volumes[[column, "geometry"]].rename(columns={column: "volume"})
    volumes = volumes.explode(index_parts=False).reset_index(drop=True)
    volumes["volume"] = pd.to_numeric(volumes["volume"], errors="coerce")

    usable = (
        volumes["volume"].notna()
        & volumes.geometry.notna()
        & ~volumes.geometry.is_empty
    )
    if not usable.all():
        logger.info("Dropping %d unusable source link(s)", int((~usable).sum()))
        volumes = volumes.loc[usable].reset_index(drop=True)
    if len(volumes) == 0:
        raise ValueError(f"{path} holds no usable volume lines.")

    logger.info(
        "Read %d source link(s) from %s, %.0f km, mean volume %.0f",
        len(volumes),
        path,
        float(volumes.to_crs(PROJECTED_CRS).geometry.length.sum()) / 1000.0,
        float(volumes["volume"].mean()),
    )
    return volumes


def carries_cars(links: gpd.GeoDataFrame) -> np.ndarray:
    """Boolean mask of the links a car may drive on."""
    return links["highway"].isin(CAR_HIGHWAYS).to_numpy(dtype=bool)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _ragged_offsets(counts: np.ndarray) -> np.ndarray:
    """``concatenate([arange(n) for n in counts])``, without the Python loop."""
    total = int(counts.sum())
    starts = np.zeros(counts.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return np.arange(total, dtype=np.int64) - np.repeat(starts, counts)


def _sample_counts(lengths: np.ndarray, spacing: float) -> np.ndarray:
    """How many samples each line gets: one per ``spacing``, and never none."""
    return np.maximum(1, np.round(lengths / spacing).astype(np.int64))


def _tangents(
    geometry: np.ndarray, position: np.ndarray, length: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit direction of each line at each position along it.

    Read over a short span rather than off the containing segment, so a sample
    that lands on a vertex takes the direction of the road rather than of
    whichever of the two segments happened to win the tie.
    """
    span = np.minimum(2.0, length / 4.0)
    before = shapely.line_interpolate_point(geometry, np.maximum(position - span, 0.0))
    after = shapely.line_interpolate_point(
        geometry, np.minimum(position + span, length)
    )
    dx = shapely.get_x(after) - shapely.get_x(before)
    dy = shapely.get_y(after) - shapely.get_y(before)
    norm = np.hypot(dx, dy)
    norm[norm == 0.0] = 1.0
    return dx / norm, dy / norm


def _samples(
    geometry: np.ndarray, lengths: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Points along the given lines, with the local direction at each.

    Samples sit at cell centres -- ``(i + 0.5) / n`` of the way along -- so none
    of them lands on an endpoint, where several roads meet and the nearest one
    says nothing about which road the line is following.
    """
    owner = np.repeat(np.arange(geometry.shape[0], dtype=np.int64), counts)
    along = (_ragged_offsets(counts) + 0.5) / counts[owner] * lengths[owner]
    points = shapely.line_interpolate_point(geometry[owner], along)
    dx, dy = _tangents(geometry[owner], along, lengths[owner])
    return points, dx, dy, owner


# --------------------------------------------------------------------------
# The match
# --------------------------------------------------------------------------


def _one_carriageway(links: gpd.GeoDataFrame) -> np.ndarray:
    """
    Links that could be one side of a divided road: one-way, and not a roundabout.

    A roundabout is one-way too and its arcs face every direction, so leaving it
    in would have every source that crosses one read as a divided road.
    """
    if "oneway" not in links.columns:
        return np.zeros(len(links), dtype=bool)
    oneway = links["oneway"].isin(ONEWAY_VALUES).to_numpy(dtype=bool, copy=True)
    if "junction" in links.columns:
        oneway &= ~(links["junction"] == "roundabout").to_numpy(dtype=bool)
    return oneway


def _claims(
    points: np.ndarray,
    sample_dx: np.ndarray,
    sample_dy: np.ndarray,
    tree: shapely.STRtree,
    target_geometry: np.ndarray,
    carriageway: np.ndarray,
    settings: MatchSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Which road each sample claims, and which road running the other way.

    Returns ``(sample, target, distance, heading)`` for the accepted claims:
    the best road for every sample that has one, plus -- where both are one-way
    -- the best road running against it, which is the other carriageway of a
    divided road. ``heading`` is +1 when the link is digitised with the source
    line and -1 when against it, which is what the divided-road test reads.
    """
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    empty_h = np.empty(0, dtype=np.int8)

    sample, target = tree.query(
        points, predicate="dwithin", distance=settings.max_distance_m
    )
    if sample.shape[0] == 0:
        return empty_i, empty_i, empty_f, empty_h

    candidate = target_geometry[target]
    candidate_length = shapely.length(candidate)
    position = shapely.line_locate_point(candidate, points[sample])
    distance = shapely.distance(points[sample], candidate)
    dx, dy = _tangents(candidate, position, candidate_length)

    alignment = sample_dx[sample] * dx + sample_dy[sample] * dy
    aligned = np.abs(alignment) >= np.cos(np.deg2rad(settings.max_angle_deg))
    if not aligned.any():
        return empty_i, empty_i, empty_f, empty_h
    sample, target, distance, alignment = (
        sample[aligned],
        target[aligned],
        distance[aligned],
        alignment[aligned],
    )

    # Rank a candidate by its distance stretched by how far off parallel it is,
    # so a road crossing at 40 degrees loses to the one the line is following
    # even when it happens to pass a little closer.
    effective = distance / np.maximum(np.abs(alignment), 1e-3)
    order = np.lexsort((effective, sample))
    sample, target, distance = sample[order], target[order], distance[order]
    heading = np.sign(alignment[order]).astype(np.int8)

    best = np.ones(sample.shape[0], dtype=bool)
    best[1:] = sample[1:] != sample[:-1]

    chosen = np.full(points.shape[0], -1, dtype=np.int64)
    chosen[sample[best]] = target[best]
    chosen_heading = np.zeros(points.shape[0], dtype=np.int8)
    chosen_heading[sample[best]] = heading[best]

    # The other carriageway: runs the other way, and both sides are one-way.
    # Two-way roads do not come in pairs, so a parallel two-way road a few
    # metres off is a different road, not this one's other half.
    claimed = chosen[sample]
    opposite = (
        (heading != chosen_heading[sample])
        & (claimed >= 0)
        & carriageway[target]
        & carriageway[np.where(claimed >= 0, claimed, 0)]
    )
    # Rows are still sorted by (sample, effective), so the first opposite row of
    # each sample is that sample's best road running the other way.
    candidates = np.flatnonzero(opposite)
    first = np.ones(candidates.shape[0], dtype=bool)
    first[1:] = sample[candidates][1:] != sample[candidates][:-1]
    second = np.zeros(sample.shape[0], dtype=bool)
    second[candidates[first]] = True

    accepted = best | second
    return sample[accepted], target[accepted], distance[accepted], heading[accepted]


def _chunks(counts: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """
    Split the source rows into blocks of roughly ``budget`` samples each.

    Never splits a source, so coverage and the divided-road test are decided from
    a whole link inside one chunk and no state has to cross a boundary.
    """
    blocks: list[tuple[int, int]] = []
    start = 0
    running = 0
    for i, n in enumerate(counts):
        running += int(n)
        if running >= budget:
            blocks.append((start, i + 1))
            start = i + 1
            running = 0
    if start < counts.shape[0]:
        blocks.append((start, counts.shape[0]))
    return blocks


def match_volumes(
    links: gpd.GeoDataFrame,
    volumes: gpd.GeoDataFrame,
    settings: MatchSettings | None = None,
) -> VolumeMatch:
    """
    Match a volume layer onto the car-carrying links of a link layer.

    Both layers are worked in :data:`~valma_bike_and_walk.config.PROJECTED_CRS`,
    so every tolerance here is metres.
    """
    settings = settings or MatchSettings()

    car = carries_cars(links)
    if not car.any():
        raise ValueError(
            "No link in this layer carries motor traffic, so no volume can be "
            f"attached to it. Expected some of {sorted(CAR_HIGHWAYS)} in 'highway'."
        )
    car_rows = np.flatnonzero(car)
    targets = links.loc[car].to_crs(PROJECTED_CRS)
    target_geometry = targets.geometry.to_numpy()
    carriageway = _one_carriageway(targets)
    tree = shapely.STRtree(target_geometry)
    logger.info(
        "%d of %d link(s) carry cars and may take a volume", len(targets), len(links)
    )

    source = volumes.to_crs(PROJECTED_CRS)
    source_geometry = source.geometry.to_numpy()
    source_volume = source["volume"].to_numpy(dtype=float)
    source_length = shapely.length(source_geometry)
    counts = _sample_counts(source_length, settings.spacing_m)

    # A national volume layer against one city's links is the normal case, so
    # the rest of the country is skipped rather than reported as a failure --
    # and not sampled at all, which is most of the work saved.
    in_extent = _within_reach(source_geometry, targets.total_bounds, settings)
    counts[~in_extent] = 0
    if not in_extent.any():
        logger.warning(
            "No source link lies within %.0f m of this link layer. Are the two "
            "covering the same place?",
            settings.max_distance_m,
        )
        return _empty_match(np.full(len(source), np.nan), in_extent)
    if not in_extent.all():
        logger.info(
            "%d of %d source link(s) lie within reach of the link layer",
            int(in_extent.sum()),
            len(source),
        )

    matched_samples = np.zeros(len(source), dtype=np.int64)
    claim_blocks: list[pd.DataFrame] = []

    with Progress(int(counts.sum()), "Matching volumes") as progress:
        for start, stop in _chunks(counts, SAMPLES_PER_CHUNK):
            block = slice(start, stop)
            points, dx, dy, owner = _samples(
                source_geometry[block], source_length[block], counts[block]
            )
            sample, target, distance, heading = _claims(
                points, dx, dy, tree, target_geometry, carriageway, settings
            )
            progress.advance(points.shape[0])
            if sample.shape[0] == 0:
                continue

            # A sample may claim two carriageways; coverage asks how much of the
            # source found *a* road, so each sample counts once.
            found = np.unique(sample)
            matched_samples[block] += np.bincount(
                owner[found], minlength=stop - start
            ).astype(np.int64)

            weight = np.exp(-distance / DISTANCE_DECAY_M)
            claim_blocks.append(
                pd.DataFrame(
                    {
                        "source": owner[sample] + start,
                        "target": target,
                        "weight": weight,
                        "forward": np.where(heading > 0, weight, 0.0),
                        "backward": np.where(heading < 0, weight, 0.0),
                        "distance": distance,
                    }
                )
                .groupby(["source", "target"], sort=False)
                .agg(
                    weight=("weight", "sum"),
                    forward=("forward", "sum"),
                    backward=("backward", "sum"),
                    distance=("distance", "mean"),
                )
                .reset_index()
            )

    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(in_extent, matched_samples / counts, np.nan)
    if not claim_blocks:
        return _empty_match(coverage, in_extent)

    claims = pd.concat(claim_blocks, ignore_index=True)
    claims = claims.loc[
        coverage[claims["source"].to_numpy()] >= settings.min_coverage
    ].reset_index(drop=True)
    if len(claims) == 0:
        return _empty_match(coverage, in_extent)

    source_used = np.zeros(len(source), dtype=bool)
    source_used[claims["source"].to_numpy()] = True
    divided = _divided_sources(claims, len(source), settings)

    # Winner takes all: a link's volume is that of the one source that claimed
    # most of it, never a blend. A stray claim from a crossing road cannot then
    # drag a link's volume halfway towards it.
    winner = (
        claims.sort_values(["target", "weight"], ascending=[True, False])
        .drop_duplicates("target", keep="first")
        .reset_index(drop=True)
    )
    source_row = winner["source"].to_numpy()
    is_divided = divided[source_row]

    return VolumeMatch(
        link_row=car_rows[winner["target"].to_numpy()],
        source_row=source_row,
        volume=np.where(
            is_divided, source_volume[source_row] / 2.0, source_volume[source_row]
        ),
        offset_m=winner["distance"].to_numpy(dtype=float),
        divided=is_divided,
        source_coverage=coverage,
        source_used=source_used,
        source_in_extent=in_extent,
    )


def _within_reach(
    source_geometry: np.ndarray,
    target_bounds: np.ndarray,
    settings: MatchSettings,
) -> np.ndarray:
    """
    Source links whose own extent comes within the tolerance of the link layer's.

    A bounding box each, not the geometry: a source that fails this cannot have a
    sample within ``max_distance_m`` of any link, and one that passes is only put
    forward as a candidate.
    """
    west, south, east, north = target_bounds
    reach = settings.max_distance_m
    bounds = shapely.bounds(source_geometry)
    return (
        (bounds[:, 0] <= east + reach)
        & (bounds[:, 2] >= west - reach)
        & (bounds[:, 1] <= north + reach)
        & (bounds[:, 3] >= south - reach)
    )


def _empty_match(coverage: np.ndarray, in_extent: np.ndarray) -> VolumeMatch:
    logger.warning("No source link matched a car-carrying link.")
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=float)
    return VolumeMatch(
        link_row=empty_i,
        source_row=empty_i,
        volume=empty_f,
        offset_m=empty_f,
        divided=np.empty(0, dtype=bool),
        source_coverage=coverage,
        source_used=np.zeros(coverage.shape[0], dtype=bool),
        source_in_extent=in_extent,
    )


def _divided_sources(
    claims: pd.DataFrame, n_sources: int, settings: MatchSettings
) -> np.ndarray:
    """
    Which sources describe both carriageways of a divided road at once.

    A source is divided when its claims land on links running both ways and
    neither direction is a rounding error -- and only one-way links can be
    claimed against the line's own direction at all, which is what makes this a
    statement about carriageways rather than about digitising.
    """
    if not settings.split_divided:
        return np.zeros(n_sources, dtype=bool)

    source = claims["source"].to_numpy()
    total = np.bincount(
        source, weights=claims["weight"].to_numpy(), minlength=n_sources
    )
    forward = np.bincount(
        source, weights=claims["forward"].to_numpy(), minlength=n_sources
    )
    backward = np.bincount(
        source, weights=claims["backward"].to_numpy(), minlength=n_sources
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        share_forward = np.where(total > 0, forward / total, 0.0)
        share_backward = np.where(total > 0, backward / total, 0.0)

    divided = (share_forward >= DIVIDED_MIN_SHARE) & (
        share_backward >= DIVIDED_MIN_SHARE
    )
    logger.info(
        "%d source link(s) read as a divided road; their volume is split between "
        "the two carriageways",
        int(divided.sum()),
    )
    return divided


# --------------------------------------------------------------------------
# Attaching the result
# --------------------------------------------------------------------------


def add_volumes(
    links: gpd.GeoDataFrame,
    volumes: gpd.GeoDataFrame,
    settings: MatchSettings | None = None,
) -> tuple[gpd.GeoDataFrame, VolumeMatch]:
    """
    Attach the car-volume columns to a link layer.

    This is the one call the pipeline makes. The match itself comes back too, so
    a caller that wants to write the source links nothing matched -- see
    :func:`unmatched_sources` -- does not have to run it twice.
    """
    match = match_volumes(links, volumes, settings)

    links = links.copy()
    volume = np.full(len(links), np.nan)
    offset = np.full(len(links), np.nan)
    volume[match.link_row] = match.volume
    offset[match.link_row] = match.offset_m
    links[VOLUME_COLUMN] = volume
    links[OFFSET_COLUMN] = offset
    if OVERRIDE_COLUMN not in links.columns:
        links[OVERRIDE_COLUMN] = np.nan

    logger.info(summarise(links, volumes, match))
    return links, match


def unmatched_sources(
    volumes: gpd.GeoDataFrame, match: VolumeMatch
) -> gpd.GeoDataFrame:
    """
    The source links whose volume went nowhere, with why, for inspection in QGIS.

    A motorway dropped from the bicycle network is the expected case and shows a
    coverage near zero. A busy road with coverage just under the threshold is the
    one worth looking at -- either the tolerance is too tight for how the two
    networks were digitised, or that road really is only half present here.

    Sources outside the link layer's extent are left out: they are what a
    national volume layer holds about everywhere else, not a match that failed.
    """
    missing = ~match.source_used & match.source_in_extent
    out = volumes.loc[missing].copy()
    out["coverage"] = match.source_coverage[missing]
    return out.reset_index(drop=True)


def summarise(
    links: gpd.GeoDataFrame, volumes: gpd.GeoDataFrame, match: VolumeMatch
) -> str:
    """
    One line per fact worth checking after a run.

    The vehicle-kilometre ratio is the summary that catches a broken match: it is
    what the matched links carry against what the sources they came from carry,
    so a match that doubled a divided road or lost half a corridor shows up as a
    number that is not near 1.
    """
    car = carries_cars(links)
    # From the geometry rather than from length_m: the same rule the rest of the
    # pipeline follows, and it holds whether or not this layer has been
    # normalised yet.
    length = geometry_lengths(links)
    matched = np.zeros(len(links), dtype=bool)
    matched[match.link_row] = True

    car_km = float(length[car].sum()) / 1000.0
    matched_km = float(length[matched].sum()) / 1000.0

    source = volumes.to_crs(PROJECTED_CRS)
    source_length = shapely.length(source.geometry.to_numpy())
    source_volume = source["volume"].to_numpy(dtype=float)
    used = match.source_used
    source_vkm = float((source_volume[used] * source_length[used]).sum()) / 1000.0
    matched_vkm = float((match.volume * length[match.link_row]).sum()) / 1000.0
    ratio = matched_vkm / source_vkm if source_vkm > 0 else float("nan")

    # Only sources over this link layer count as offered; the rest of a national
    # volume layer was never in the running.
    offered = match.source_in_extent
    lost = offered & ~used
    lost_vkm = float((source_volume[lost] * source_length[lost]).sum()) / 1000.0

    return (
        f"Volumes attached to {match.n_links:,} link(s), {matched_km:,.0f} km of "
        f"{car_km:,.0f} km that carry cars ({matched_km / car_km:.0%}); "
        f"{int(used.sum()):,} of {int(offered.sum()):,} source link(s) over this "
        f"extent used, {int(match.divided.sum()):,} link(s) on a divided road "
        f"took half a source's volume; vehicle-km carried across {ratio:.0%}, "
        f"{lost_vkm:,.0f} veh-km on source links with no road here"
    )


__all__ = [
    "CAR_HIGHWAYS",
    "DEFAULT_MAX_ANGLE_DEG",
    "DEFAULT_MAX_DISTANCE_M",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_SPACING_M",
    "DEFAULT_VOLUME_COLUMN",
    "MatchSettings",
    "OFFSET_COLUMN",
    "OVERRIDE_COLUMN",
    "VOLUME_COLUMN",
    "VolumeMatch",
    "add_volumes",
    "carries_cars",
    "match_volumes",
    "read_volumes",
    "summarise",
    "unmatched_sources",
]
