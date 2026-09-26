"""Pipeline step 8 -> data/demo_cache.json + data/demo_images/*.png. Owner: D.

For config.demo.n_curated_base_images base images x each MVP corruption x severities 1-5, plus clean:
build a schemas.PredictResponse. PNGs are nearest-neighbor upscaled 32 -> 256 px, named <sample_id>.png,
served at /images/<sample_id>.png.
"""
from __future__ import annotations

from common import cli

STEP, OWNER = "build_demo_cache", "D"


def pick_curated_base_images(features_df, n: int) -> list[str]:
    """TODO(D): curation rule (e.g., mix of images that fail/survive under corruption)."""
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    cli.not_implemented(STEP, OWNER)


if __name__ == "__main__":
    main()
