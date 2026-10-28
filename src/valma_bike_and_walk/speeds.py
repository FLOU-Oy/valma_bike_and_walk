"""Travel speeds for walking and cycling.

OSMnx's ``add_edge_speeds`` imputes speeds from the OSM ``maxspeed`` tag, which
records the *motor traffic* speed limit. Using it for a bike or pedestrian
network makes a cyclist do 50 km/h down an arterial and says nothing at all
about the surface underfoot, so travel times come out far too optimistic.

Instead we derive speed from the two tags that actually govern how fast a person
moves: what kind of way it is (``highway``) and what it is made of (``surface``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600.0
TRAFFIC_COEFFICIENT = 0.014

#: Tarkkala et al. (2024) estimate the traffic-volume term once per *route*
#: (veh/h, not multiplied by length), unlike the per-km road-class and facility
#: terms. Spreading it over their survey's average route length (Table 5) turns
#: it into a per-km term, added to the link's road-class + facility disutility
#: (:func:`traffic_factor`).
AVG_ROUTE_LENGTH_KM = 5.6

#: Tarkkala et al. (2024), Table 4 (M1): a route's disutility per km is the
#: *sum* of a road-class and a bike-facility coefficient. The facility
#: coefficients are kept here as multiples of the separated-path one -- the
#: cycle-track factor where the road has a kerb-separated track
#: (``cycleway=track``), the cycle-lane factor where it has a lane, the
#: mixed-traffic factor otherwise. M1's facility coefficients are mixed 5.03,
#: lane 2.98, adjacent cycle path 2.52 and separated 2.61; a track beside the
#: road is their *adjacent* path, a hair better than a separated one.
BETA_SEPARATED = 2.61
MIXED_TRAFFIC_FACTOR = 1.927
CYCLE_LANE_FACTOR = 1.142
CYCLE_TRACK_FACTOR = 0.966

#: Road-class coefficients per km (M1): local covers residential and collector
#: streets, main the main streets, arterial the arterial roads. ``unclassified``
#: and ``service`` are left out: they carry so little traffic that riding them
#: counts as a separated path on a local link, the reference a cycleway stands
#: at, not as mixed traffic. A car volume where one is known still costs on top.
BETA_LOCAL = 4.96
BETA_MAIN = 4.14
BETA_ARTERIAL = 4.05
ROAD_CLASS_BETA: Mapping[str, float] = {
    "living_street": BETA_LOCAL,
    "residential": BETA_LOCAL,
    "tertiary": BETA_LOCAL,
    "tertiary_link": BETA_LOCAL,
    "secondary": BETA_MAIN,
    "secondary_link": BETA_MAIN,
    "primary": BETA_ARTERIAL,
    "primary_link": BETA_ARTERIAL,
}

#: The reference the cycleway's speed stands for: a separated path on a local
#: link, which is what Tarkkala coded a path away from main roads as. Every
#: path stands here, at its own nominal speed. A road's bike speed is its
#: nominal speed scaled by how much more its own road class + facility cost
#: per km than this -- never by less than 1: a main street with a track
#: (4.14 + 2.52) scores below the reference, and riding it is preferred, not
#: quicker than its nominal speed.
REFERENCE_DISUTIL = BETA_LOCAL + BETA_SEPARATED


#: Paths that pedestrians share unless they are split into a cycling half and
#: a walking half (``segregated=yes``). Shared and untagged paths ride at the
#: highway's own speed; a segregated one is a little quicker, with no weaving
#: round people on foot.
SHARED_PATH_HIGHWAYS = frozenset({"cycleway", "path"})
SEGREGATED_PATH_FACTOR = 1.05
SHARED_PATH_FACTOR = 1.0


def segregation_factor(highway: pd.Series, segregated: pd.Series | None) -> np.ndarray:
    """Bike speed factor for shared vs segregated foot-and-cycle paths."""
    on_path = highway.isin(SHARED_PATH_HIGHWAYS).to_numpy(dtype=bool)
    if segregated is None:
        split = np.zeros(len(highway), dtype=bool)
    else:
        split = segregated.eq("yes").fillna(False).to_numpy(dtype=bool)
    return np.where(
        on_path, np.where(split, SEGREGATED_PATH_FACTOR, SHARED_PATH_FACTOR), 1.0
    )


def link_disutility(highway: pd.Series, facility_factor: np.ndarray) -> np.ndarray:
    """
    Per-km road-class + facility disutility of each link, before traffic.

    A road outside :data:`ROAD_CLASS_BETA` (a path, a service road) has no
    carriageway cost of its own and stands at :data:`REFERENCE_DISUTIL`, so
    its speed is not divided by anything until traffic is added to it.
    """
    road = highway.map(ROAD_CLASS_BETA).astype(float).to_numpy()
    return np.where(
        np.isfinite(road),
        np.nan_to_num(road) + BETA_SEPARATED * np.asarray(facility_factor, dtype=float),
        REFERENCE_DISUTIL,
    )


def traffic_factor(disutility: np.ndarray, car_volume: pd.Series | np.ndarray) -> np.ndarray:
    """
    Speed multiplier for car traffic, 1 where the volume is missing.

    The traffic term joins the same per-km sum as road class and facility,
    so a link's whole divisor is ``(road + facility + traffic) /
    REFERENCE_DISUTIL``: this is that sum without traffic over the sum with it.
    (Where road + facility alone score below the reference, the carriageway
    part of the divisor is held at 1 and this factor applies on top.)
    """
    volume = pd.to_numeric(pd.Series(np.asarray(car_volume)), errors="coerce").to_numpy(
        dtype=float
    )
    term = TRAFFIC_COEFFICIENT * np.nan_to_num(volume) / AVG_ROUTE_LENGTH_KM
    return np.where(np.isfinite(volume), disutility / (disutility + term), 1.0)


def carriageway_divisor(highway: str, facility_factor: float) -> float:
    """What a road's nominal bike speed is divided by: its road-class plus
    facility disutility per km, relative to :data:`REFERENCE_DISUTIL`."""
    road = ROAD_CLASS_BETA[highway]
    return max((road + BETA_SEPARATED * facility_factor) / REFERENCE_DISUTIL, 1.0)


@dataclass(frozen=True)
class SpeedProfile:
    """Maps OSM way tags onto a travel speed in km/h."""

    name: str
    base_kph: float
    highway_kph: Mapping[str, float]
    surface_factor: Mapping[str, float]
    min_kph: float
    max_kph: float

    def speeds_kph(
        self,
        highway: pd.Series,
        surface: pd.Series | None = None,
        segregated: pd.Series | None = None,
        cycleway: pd.Series | None = None,
        car_volume: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Vectorised speed lookup for a whole edge table.

        Unknown or missing tags fall back to ``base_kph`` and a surface factor
        of 1.0, so an edge is never left without a usable speed.
        """
        base = highway.map(self.highway_kph).astype(float).fillna(self.base_kph)

        disutility = None
        if self.name == "bike":
            if cycleway is None:
                track_mask = lane_mask = np.zeros(len(highway), dtype=bool)
            else:
                # ";"-separated values: the best one present wins.
                values = cycleway.fillna("").astype(str).str.lower()
                track_mask = values.str.contains(
                    r"(?:^|;)\s*track\s*(?:;|$)", regex=True
                ).to_numpy(dtype=bool)
                lane_mask = values.str.contains(
                    r"(?:^|;)\s*lane\s*(?:;|$)", regex=True
                ).to_numpy(dtype=bool)
            facility_factor = np.select(
                [track_mask, lane_mask],
                [CYCLE_TRACK_FACTOR, CYCLE_LANE_FACTOR],
                default=MIXED_TRAFFIC_FACTOR,
            )
            disutility = link_disutility(highway, facility_factor)
            base = base / np.maximum(disutility / REFERENCE_DISUTIL, 1.0)

        if surface is None:
            factor: pd.Series | float = 1.0
        else:
            factor = surface.map(self.surface_factor).astype(float).fillna(1.0)

        if self.name == "bike":
            factor = factor * segregation_factor(highway, segregated)

        if car_volume is not None and disutility is not None:
            factor = factor * pd.Series(
                traffic_factor(disutility, car_volume), index=highway.index
            )

        return np.clip(base * factor, self.min_kph, self.max_kph).to_numpy(dtype=float)

    def travel_times_seconds(
        self,
        length_m: np.ndarray,
        highway: pd.Series,
        surface: pd.Series | None = None,
        segregated: pd.Series | None = None,
        car_volume: pd.Series | None = None,
    ) -> np.ndarray:
        """Seconds to traverse each edge, given its length in metres."""
        speed_ms = self.speeds_kph(highway, surface, segregated, car_volume=car_volume) * (
            1000.0 / SECONDS_PER_HOUR
        )
        return np.asarray(length_m, dtype=float) / speed_ms


# Walking pace is close to constant on anything flat and paved; what actually
# changes it is stairs and rough ground. ~4.8 km/h is the usual adult average.
WALK_PROFILE = SpeedProfile(
    name="walk",
    base_kph=4.8,
    highway_kph={
        "steps": 1.2,
        "elevator": 1.0,
        "path": 4.3,
        "track": 4.3,
        "bridleway": 4.0,
        "corridor": 4.5,
        "footway": 4.8,
        "pedestrian": 4.8,
        "living_street": 4.8,
        "residential": 4.8,
        "service": 4.8,
        "cycleway": 4.8,
        "unclassified": 4.8,
        "tertiary": 4.8,
        "tertiary_link": 4.8,
        "secondary": 4.8,
        "secondary_link": 4.8,
        "primary": 4.7,
        "primary_link": 4.7,
        "trunk": 4.5,
        "trunk_link": 4.5,
    },
    surface_factor={
        "asphalt": 1.0,
        "concrete": 1.0,
        "paved": 1.0,
        "paving_stones": 0.98,
        "compacted": 0.98,
        "fine_gravel": 0.97,
        "wood": 0.95,
        "gravel": 0.92,
        "cobblestone": 0.90,
        "sett": 0.90,
        "unpaved": 0.92,
        "ground": 0.88,
        "dirt": 0.88,
        "earth": 0.88,
        "grass": 0.85,
        "snow": 0.80,
        "ice": 0.75,
        "sand": 0.75,
        "mud": 0.70,
    },
    min_kph=0.8,
    max_kph=6.0,
)

# Cycling is far more sensitive to both way type and surface: a segregated
# cycleway is quick, a muddy forest track is barely faster than walking, and
# steps mean dismounting and pushing.
BIKE_PROFILE = SpeedProfile(
    name="bike",
    base_kph=14.0,
    # Roads in ROAD_CLASS_BETA are nominal speeds here, divided by their
    # road-class + facility cost relative to a separated path on a local link
    # (carriageway_divisor, applied in speeds_kph).
    highway_kph={
            "steps": 1.2,
            "elevator": 1.0,
            "cycleway": 18.0,
            "path": 11.0,
            "track": 10.0,
            "bridleway": 8.0,
            "footway": 4.0,
            "pedestrian": 6.0,
            "corridor": 5.0,
            "living_street": 13.0,
            "residential": 18.0,
            "service": 16.0,
            "unclassified": 18.0,
            "tertiary": 18.0,
            "tertiary_link": 16.0,
            "secondary": 18.0,
            "secondary_link": 16.0,
            "primary": 18.0,
            "primary_link": 16.0,
            "trunk": 18.0, 
            "trunk_link": 16.0,
        },
    surface_factor={
        "asphalt": 1.0,
        "concrete": 1.0,
        "paved": 1.0,
        "paving_stones": 0.90,
        # Stone dust (kivituhka) rides close to asphalt in summer, but these
        # paths get lighter winter maintenance; set for the whole year.
        "compacted": 0.87,
        "fine_gravel": 0.87,
        "wood": 0.80,
        "gravel": 0.70,
        "unpaved": 0.70,
        "cobblestone": 0.65,
        "sett": 0.65,
        "ground": 0.60,
        "dirt": 0.60,
        "earth": 0.60,
        "grass": 0.50,
        "snow": 0.45,
        "sand": 0.40,
        "mud": 0.40,
        "ice": 0.35,
    },
    min_kph=1.0,
    max_kph=30.0,
)

PROFILES: dict[str, SpeedProfile] = {
    WALK_PROFILE.name: WALK_PROFILE,
    BIKE_PROFILE.name: BIKE_PROFILE,
}


def profile_for(mode: str) -> SpeedProfile:
    try:
        return PROFILES[mode]
    except KeyError:
        raise ValueError(
            f"No speed profile for mode {mode!r}; have {sorted(PROFILES)}."
        ) from None
