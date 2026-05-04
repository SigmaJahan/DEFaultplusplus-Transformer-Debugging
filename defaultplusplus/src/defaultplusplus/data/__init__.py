"""Public bench-dataset distribution.

The training data CSVs are too large to ship in the wheel, so the
package downloads them on demand.

    from defaultplusplus.data import download_bench
    path = download_bench(version="v1")          # ~/.cache/defaultplusplus/bench/v1
    # or
    defaultpp-bench-download                     # console script

Verifies SHA256 before extraction and short-circuits when an existing
download already matches the expected checksum, so re-running is free.
"""
from .download import (
    BENCH_VERSIONS,
    BenchVersion,
    DatasetNotPublishedError,
    DownloadError,
    bench_dir,
    download_bench,
)

__all__ = [
    "BENCH_VERSIONS",
    "BenchVersion",
    "DatasetNotPublishedError",
    "DownloadError",
    "bench_dir",
    "download_bench",
]
