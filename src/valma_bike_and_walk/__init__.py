"""Bike and walking travel-time analysis from a local OpenStreetMap extract."""

from valma_bike_and_walk.config import MODES, PROJECTED_CRS, WGS84, Settings
from valma_bike_and_walk.network import RoutableNetwork, build_network, load_network
from valma_bike_and_walk.speeds import BIKE_PROFILE, WALK_PROFILE, SpeedProfile

__all__ = [
    "BIKE_PROFILE",
    "MODES",
    "PROJECTED_CRS",
    "RoutableNetwork",
    "Settings",
    "SpeedProfile",
    "WALK_PROFILE",
    "WGS84",
    "build_network",
    "load_network",
]

__version__ = "0.2.0"
