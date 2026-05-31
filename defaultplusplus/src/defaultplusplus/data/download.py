"""Bench dataset download + checksum verification.

The function is idempotent: if the cached extract already matches the
expected SHA256 we skip the download entirely. Only the tarball is
hashed during transfer; per-file digests are checked against the
in-bundle ``MANIFEST.sha256`` after extraction.

Adding a new version is one entry in :data:`BENCH_VERSIONS`. Once you
publish to Zenodo:

    1. Run ``data/stage_release_bundle.py`` to produce
       ``dist/defaultpp-bench-v<n>.tar.gz`` and its ``.sha256`` sidecar.
    2. Upload the tarball to Zenodo, mint a DOI, copy the
       direct-download URL.
    3. Paste the URL into the ``url`` field below; paste the SHA256
       (already in the sidecar) into ``sha256``. Bump
       ``DEFAULT_VERSION`` if this is the new latest.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BenchVersion:
    """One entry in the published-versions table.

    ``url`` is None when the version exists locally but has not been
    uploaded yet; ``download_bench`` raises :class:`DatasetNotPublishedError`
    in that case so callers get a clear error rather than a 404.
    """
    name: str
    url: Optional[str]
    sha256: str
    license: str
    description: str


# Published versions. Each entry pins the direct-download URL Zenodo
# returns for the tarball under that record's DOI, plus the SHA256 the
# stage_release_bundle.py script printed when the tarball was built.
# Adding a new version is one entry here.
BENCH_VERSIONS: dict[str, BenchVersion] = {
    "v1": BenchVersion(
        name="v1",
        url="https://zenodo.org/records/20018623/files/defaultpp-bench-v1.tar.gz",
        sha256="da63b9b52c58011ec3423faf4d0037f6d2e8a575230391c5572929a6f2be2cb3",
        license="CC-BY-4.0",
        description=(
            "DEFault++ benchmark v1 — 6,042 encoder rows + 2,535 decoder "
            "rows of paper-aligned feature CSVs covering 35 model-task "
            "fine-tunes. Includes per-task source CSVs, merged "
            "trainer-ready CSVs, feature dictionary, and integrity "
            "manifest. Synthetic-zero padding marked with the "
            "``__synthetic_zero`` suffix; see README inside the bundle. "
            "DOI: 10.5281/zenodo.20018623"
        ),
    ),
}

DEFAULT_VERSION = "v1"


class DownloadError(RuntimeError):
    """Network failure, checksum mismatch, or tar extraction failure."""


class DatasetNotPublishedError(DownloadError):
    """The requested version exists in the version table but has no
    download URL yet (``url=None``).

    This is the normal state for a freshly-staged release that has not
    been uploaded to Zenodo. Build the tarball with
    ``data/stage_release_bundle.py`` and upload it before ``url`` can be
    filled in.
    """


# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
def _cache_root() -> Path:
    """Return ``$XDG_CACHE_HOME/defaultplusplus`` (or platform default).

    Falls back to ``~/.cache/defaultplusplus`` on POSIX and
    ``~/AppData/Local/defaultplusplus/cache`` on Windows.
    """
    env = os.environ.get("DEFAULTPP_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "defaultplusplus"
    if sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA") or "~/AppData/Local"
        return Path(appdata).expanduser().resolve() / "defaultplusplus" / "cache"
    return Path("~/.cache/defaultplusplus").expanduser().resolve()


def bench_dir(version: str = DEFAULT_VERSION) -> Path:
    """Return the local extract directory for ``version`` (creates parents)."""
    return _cache_root() / "bench" / version


# ─────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────
def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(extract_root: Path) -> None:
    """Cross-check every file under ``extract_root`` against MANIFEST.sha256.

    The manifest covers all bundle files except itself; an extra or
    missing file raises. This catches partial extracts and tampering.
    """
    manifest = extract_root / "MANIFEST.sha256"
    if not manifest.exists():
        raise DownloadError(
            f"missing MANIFEST.sha256 inside {extract_root}; bundle is corrupt"
        )
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        if not digest or not rel:
            raise DownloadError(f"malformed line in MANIFEST.sha256: {line!r}")
        expected[rel] = digest

    bad: list[str] = []
    for rel, want in expected.items():
        target = extract_root / rel
        if not target.exists():
            bad.append(f"missing: {rel}")
            continue
        got = _sha256_file(target)
        if got != want:
            bad.append(f"sha256 mismatch: {rel}")
    if bad:
        raise DownloadError(
            "manifest verification failed:\n  " + "\n  ".join(bad[:10])
            + ("..." if len(bad) > 10 else "")
        )


# ─────────────────────────────────────────────────────────────────────────
# Download + extract
# ─────────────────────────────────────────────────────────────────────────
def _download_to(url: str, dest: Path, *, progress: bool = True) -> None:
    """Stream ``url`` to ``dest``, printing a tiny progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            so_far = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    so_far += len(chunk)
                    if progress and total > 0:
                        pct = 100 * so_far / total
                        sys.stderr.write(
                            f"\r[download] {so_far / 1e6:7.1f} / "
                            f"{total / 1e6:7.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stderr.flush()
        if progress:
            sys.stderr.write("\n")
        tmp.replace(dest)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to fetch {url}: {exc}") from exc


def _existing_extract_root(root: Path) -> Optional[Path]:
    """Return any directly-nested directory under ``root`` that holds a
    MANIFEST.sha256, or None when no candidate exists."""
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and (child / "MANIFEST.sha256").is_file():
            return child
    return None


def _extract_tarball(tarball: Path, dest_root: Path) -> Path:
    """Safely extract ``tarball`` into ``dest_root``, return extract root.

    Refuses any member with an absolute path or ``..`` segment so a
    malicious tarball can't escape ``dest_root``.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    extract_root: Optional[Path] = None
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            mname = member.name
            if mname.startswith("/") or ".." in Path(mname).parts:
                raise DownloadError(
                    f"refusing to extract unsafe path: {mname!r}"
                )
            if extract_root is None:
                # Top-level entry is the bundle directory.
                extract_root = dest_root / Path(mname).parts[0]
        tar.extractall(dest_root)
    if extract_root is None:
        raise DownloadError(f"empty tarball {tarball}")
    return extract_root


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────
def download_bench(
    version: str = DEFAULT_VERSION,
    *,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    url_override: Optional[str] = None,
    sha256_override: Optional[str] = None,
    verify_manifest: bool = True,
    progress: bool = True,
) -> Path:
    """Download and extract the DEFault++ benchmark for ``version``.

    Args:
        version:         entry in :data:`BENCH_VERSIONS`. Default
                         :data:`DEFAULT_VERSION`.
        cache_dir:       override the cache root. By default uses
                         ``$DEFAULTPP_CACHE_DIR`` / ``$XDG_CACHE_HOME`` /
                         platform default.
        force:           wipe any existing extract and re-download.
        url_override:    pull from this URL instead of the published
                         entry. Useful for testing against a local
                         ``file://`` URL or a private mirror.
        sha256_override: expected hash for ``url_override``. Required
                         when overriding URL.
        verify_manifest: cross-check every extracted file against
                         ``MANIFEST.sha256``. Default True; flip off for
                         quick tests on a known-good cache.
        progress:        print a one-line progress indicator to stderr.

    Returns:
        Absolute path to the extracted bundle directory (e.g.
        ``~/.cache/defaultplusplus/bench/v1/defaultpp-bench-v1``).

    Raises:
        :class:`KeyError`              unknown ``version``.
        :class:`DatasetNotPublishedError` version exists but has no URL.
        :class:`DownloadError`         network or checksum failure.
    """
    if version not in BENCH_VERSIONS:
        raise KeyError(
            f"unknown bench version {version!r}; "
            f"available: {sorted(BENCH_VERSIONS)}"
        )
    info = BENCH_VERSIONS[version]
    url = url_override or info.url
    expected_sha = sha256_override or info.sha256

    if url is None:
        raise DatasetNotPublishedError(
            f"benchmark {version!r} has not been uploaded yet. "
            f"Build the tarball with ``python data/stage_release_bundle.py`` "
            f"and upload to Zenodo. Once published, paste the direct-download "
            f"URL into BENCH_VERSIONS[{version!r}].url. To download from a "
            f"custom location now, pass ``url_override=...`` and "
            f"``sha256_override=...``."
        )

    root = (cache_dir or _cache_root()) / "bench" / version
    extract_root = root / f"defaultpp-bench-{version}"
    tarball = root / f"defaultpp-bench-{version}.tar.gz"

    # ``force=True`` wipes the entire version directory so we re-download
    # AND re-extract from scratch. We can't only wipe ``extract_root``
    # because the tarball's top-level dir name may differ from the
    # canonical ``defaultpp-bench-{version}`` (e.g. private test bundles).
    if force and root.exists():
        shutil.rmtree(root)

    # Short-circuit: cached extract is intact and matches the manifest.
    # The tarball's top-level dir might not match the canonical name
    # (e.g. test bundles, private mirrors), so probe for any directory
    # under ``root`` that contains a MANIFEST.sha256.
    if not force and root.is_dir():
        candidate = _existing_extract_root(root)
        if candidate is not None:
            try:
                if verify_manifest:
                    _verify_manifest(candidate)
                return candidate
            except DownloadError:
                # Cache is corrupt; fall through to re-download.
                shutil.rmtree(candidate, ignore_errors=True)

    # If a tarball is already on disk and hashes correctly, skip the network.
    if tarball.exists():
        got = _sha256_file(tarball)
        if got != expected_sha:
            tarball.unlink()

    if not tarball.exists():
        _download_to(url, tarball, progress=progress)

    got = _sha256_file(tarball)
    if got != expected_sha:
        tarball.unlink(missing_ok=True)
        raise DownloadError(
            f"sha256 mismatch on download from {url}: "
            f"expected {expected_sha[:12]}…, got {got[:12]}…"
        )

    # Clear any pre-existing extract under ``root`` (besides the tarball
    # itself) so a stale extract from a previous run can't shadow the
    # fresh content.
    for child in root.iterdir():
        if child == tarball:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    actual_root = _extract_tarball(tarball, root)
    if actual_root != extract_root:
        # Tarball top-level directory name differs from convention; use it.
        extract_root = actual_root
    if verify_manifest:
        _verify_manifest(extract_root)
    return extract_root
