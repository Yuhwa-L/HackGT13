"""Pipeline step 1 -> data/manifest.csv (schema: common.schemas.MANIFEST_COLUMNS). Owner: B.

Rows: CIFAR-10 test (clean) + CIFAR-10-C (config.benchmark.cifar10c_types x severities) + CIFAR-10.1.
sample_id formats: c10_00123 | c10c_<type>_00123_s3 | c101_00042.
"""
from __future__ import annotations

from common import cli, io

STEP, OWNER = "make_manifest", "B"


def build_manifest(cfg, limit: int | None):
    """Return a DataFrame matching MANIFEST_COLUMNS, with splits from make_splits.assign_splits.

    TODO(B): decide n_base_images subsampling (config.benchmark.n_base_images), then implement.
    """
    raise NotImplementedError


def main() -> None:
    args, cfg = cli.parse(cli.base_parser(__doc__))
    if args.mock:
        print(f"[{STEP}] --mock: using mock manifest from `make mock`; nothing to do.")
        return
    try:
        df = build_manifest(cfg, args.limit)
    except NotImplementedError:
        cli.not_implemented(STEP, OWNER)
    io.write_table(df, io.data_dir() / io.MANIFEST)


if __name__ == "__main__":
    main()
