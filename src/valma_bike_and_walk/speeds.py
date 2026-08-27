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
    ) -> np.ndarray:
        """
        Vectorised speed lookup for a whole edge table.

        Unknown or missing tags fall back to ``base_kph`` and a surface factor
        of 1.0, so an edge is never left without a usable speed.
        """
        base = highway.map(self.highway_kph).astype(float).fillna(self.base_kph)

        if surface is None:
            factor: pd.Series | float = 1.0
        else:
            factor = surface.map(self.surface_factor).astype(float).fillna(1.0)

        if segregated is not None and self.name == "bike":
            factor = factor * segregated.eq("yes").astype(float).mul(0.10).add(1.0)

        return np.clip(base * factor, self.min_kph, self.max_kph).to_numpy(dtype=float)

    def travel_times_seconds(
        self,
        length_m: np.ndarray,
        highway: pd.Series,
        surface: pd.Series | None = None,
        segregated: pd.Series | None = None,
    ) -> np.ndarray:
        """Seconds to traverse each edge, given its length in metres."""
        speed_ms = self.speeds_kph(highway, surface, segregated) * (
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
    highway_kph={
        "steps": 1.2,
        "elevator": 1.0,
        "cycleway": 20.0,
        "path": 11.0,
        "track": 10.0,
        "bridleway": 8.0,
        "footway": 4.0,
        "pedestrian": 6.0,
        "corridor": 5.0,
        "living_street": 13.0 / 1.927 / 1.225,
        "residential": 18.0 / 1.927 / 1.225,
        "service": 13.0 / 1.927 / 1.225,
        "unclassified": 15.0 / 1.927 / 1.225,
        "tertiary": 18.0 / 1.927 / 1.225,
        "tertiary_link": 15.0 / 1.927 / 1.225,
        "secondary": 18.0 / 1.927 / 1.022,
        "secondary_link": 16.0 / 1.927 / 1.022,
        "primary": 18.0 / 1.927,
        "primary_link": 16.0 / 1.927,
        "trunk": 18.0,
        "trunk_link": 16.0,
    },
    surface_factor={
        "asphalt": 1.0,
        "concrete": 1.0,
        "paved": 1.0,
        "paving_stones": 0.90,
        "compacted": 0.85,
        "fine_gravel": 0.80,
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
