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
FINLAND_BOUNDS = (19.0, 59.7, 31.7, 70.1)

#: Where osmium keeps the node coordinates it needs to give ways a geometry.
#:
#: "flex_mem" holds them in memory and grows as needed -- right for anything up
#: to a country. For a planet-sized file, point this at a file-backed index
#: instead, e.g. "sparse_file_array,/tmp/nodes.idx".
DEFAULT_INDEX_STORAGE = "flex_mem"


@dataclass(frozen=True)
class Settings:
    """Where things live on disk, and how the PBF should be read."""

    #: The .osm.pbf to read. Only needed by the `extract` stage; a build that
    #: starts from an already-extracted links GeoPackage leaves it None.
    pbf_path: Path | None = None
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("output")

    #: osmium node-location index, see :data:`DEFAULT_INDEX_STORAGE`.
    index_storage: str = DEFAULT_INDEX_STORAGE

    def __post_init__(self) -> None:
        if self.pbf_path is not None:
            object.__setattr__(self, "pbf_path", Path(self.pbf_path))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def source_stem(self) -> str:
        """Name the cache files are keyed on."""
        return self.pbf_path.stem if self.pbf_path is not None else "links"

    def require_pbf(self) -> Path:
        if self.pbf_path is None:
            raise ValueError("This step needs a .osm.pbf; none was configured.")
        return self.pbf_path

    def network_path(self, mode: str, links_path: Path | str) -> Path:
        """
        Where the graph built from a link layer goes: next to the results.

        Named after the layer it came from, so `output/walk_links.gpkg` builds
        `output/walk.npz`. The mode is added when the name does not already
        carry it, so two modes off one hand-named layer cannot collide.
        """
        stem = Path(links_path).stem.removesuffix("_links")
        if mode not in stem:
            stem = f"{stem}_{mode}"
        return self.output_dir / f"{stem}.npz"

    def links_cache_path(self, mode: str, extent_key: str) -> Path:
        """Where a `--pbf` run parks the link layer nobody asked for by name."""
        return self.cache_dir / f"{self.source_stem}_{extent_key}_{mode}_links.gpkg"

    def network_cache_path(self, mode: str, extent_key: str) -> Path:
        """The same, for the graph built from it."""
        return self.cache_dir / f"{self.source_stem}_{extent_key}_{mode}.npz"

    def boundary_cache_path(self, area: str) -> Path:
        safe = area.split(",")[0].strip().replace(" ", "_")
        return self.cache_dir / f"{self.source_stem}_{safe}_boundary.geojson"


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {list(MODES)}.")
    return mode
