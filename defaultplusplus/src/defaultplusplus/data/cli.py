"""``defaultpp-bench-download`` console script.

Downloads + verifies + extracts the published DEFault++ benchmark to
the user's cache. Idempotent — re-running on a verified cache is a
no-op.

    defaultpp-bench-download              # latest version
    defaultpp-bench-download --version v1
    defaultpp-bench-download --force      # bypass cache and re-fetch
    defaultpp-bench-download --list       # show known versions

Override the source for testing or mirroring:

    defaultpp-bench-download --url file:///path/to/bundle.tar.gz \\
                             --sha256 <expected sha>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .download import (
    BENCH_VERSIONS,
    DEFAULT_VERSION,
    DatasetNotPublishedError,
    DownloadError,
    bench_dir,
    download_bench,
)


def _format_versions() -> str:
    rows = []
    for name, info in BENCH_VERSIONS.items():
        status = "PUBLISHED" if info.url else "NOT YET PUBLISHED"
        rows.append(f"  {name:6s} [{status}]  {info.description}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help=f"bench version (default: {DEFAULT_VERSION})")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="override the cache root (default: "
                        "$DEFAULTPP_CACHE_DIR / $XDG_CACHE_HOME / platform default)")
    p.add_argument("--force", action="store_true",
                   help="wipe cached extract and re-download")
    p.add_argument("--url", default=None,
                   help="custom source URL (file:// or http(s)://); requires --sha256")
    p.add_argument("--sha256", default=None,
                   help="expected SHA256 for --url")
    p.add_argument("--no-verify", action="store_true",
                   help="skip per-file MANIFEST verification (faster on known-good cache)")
    p.add_argument("--list", action="store_true",
                   help="print the known bench versions and exit")
    args = p.parse_args(argv)

    if args.list:
        print("Available bench versions:")
        print(_format_versions())
        return 0

    if (args.url is None) ^ (args.sha256 is None):
        print("ERROR: --url and --sha256 must be used together",
              file=sys.stderr)
        return 2

    try:
        path = download_bench(
            version=args.version,
            cache_dir=args.cache_dir,
            force=args.force,
            url_override=args.url,
            sha256_override=args.sha256,
            verify_manifest=not args.no_verify,
        )
    except DatasetNotPublishedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] benchmark {args.version} ready at:")
    print(f"  {path}")
    print()
    print(f"Trainer entry point:")
    print(f"  python defaultplusplus/scripts/train_diagnoser.py \\\\")
    print(f"      --arch encoder \\\\")
    print(f"      --csv {path / 'encoder_merged.csv'} \\\\")
    print(f"      --output encoder.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
