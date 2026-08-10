"""Shared configuration: coordinate systems, mode names and default paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Coordinate system OSM data is stored in.
WGS84 = "EPSG:4326"

#: Finnish national grid. Metric, so lengths and snapping distances are metres.
PROJECTED_CRS = "EPSG:3067"

#: Travel modes this project supports, in our own vocabulary.
MODES = ("walk", "bike")

#: Mainland Finland plus Åland, as [min_lon, min_lat, max_lon, max_lat].
#: The default extent for a tiled country-wide build.
FINLAND_BOUNDS = (19.0, 59.7, 31.7, 70.1)

#: Our mode names mapped onto the network names pyrosm understands.
PYROSM_NETWORK_TYPES = {
    "walk": "walking",
    "bike": "cycling",
}


@dataclass(frozen=True)
class Settings:
    """Where things live on disk, and how the PBF should be read."""

    pbf_path: Path
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("output")

    #: pyrosm reader engine, or "auto" to let the build decide.
    #:
    #: "out_of_core" decodes the PBF into temporary shard files and streams them
    #: back, which keeps memory flat and uses every core -- but it writes the
    #: whole decoded file to disk on *every* read (~1.3 GB of spill for a 190 MB
    #: extract). One big read: worth it. Sixteen concurrent tiles: sixteen
    #: simultaneous spills, and the disk becomes the bottleneck.
    #:
    #: "in_memory" is pyrosm's own default. Single-core and it holds everything
    #: in RAM, but it touches the disk only to read the file.
    #:
    #: "auto" picks in_memory for parallel tiled builds and out_of_core
    #: otherwise.
    engine: str = "auto"

    #: Parse workers: an integer, "auto" for one per core, or None for one.
    #: Only meaningful with the out_of_core engine.
    osm_workers: int | str | None = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "pbf_path", Path(self.pbf_path))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def network_cache_path(
        self, mode: str, extent_key: str, geometry: bool = False
    ) -> Path:
        # Geometry-carrying builds get their own cache file so a plain build
        # never silently loads (or is silently shadowed by) a --gpkg one, and
        # vice versa.
        suffix = "_geom" if geometry else ""
        return self.cache_dir / f"{self.pbf_path.stem}_{extent_key}_{mode}{suffix}.npz"

    def boundary_cache_path(self, area: str) -> Path:
        safe = area.split(",")[0].strip().replace(" ", "_")
        return self.cache_dir / f"{self.pbf_path.stem}_{safe}_boundary.geojson"


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {list(MODES)}.")
    return mode
