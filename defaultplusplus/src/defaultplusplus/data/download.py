"""Bench dataset download + checksum verification.

The function is idempotent: if the cached extract already exists it is
returned immediately. Downloads are SHA256-verified when ``sha256`` is
non-empty; leave it as ``""`` to skip verification (fill it in after
uploading so users get integrity checking).

Adding a new version is one entry in :data:`BENCH_VERSIONS`. Once you
publish to Zenodo:

    1. Upload the zip to Zenodo, mint a DOI, copy the direct-download URL.
    2. Paste the URL into the ``url`` field below.
    3. Run ``shasum -a 256 <downloaded-file>.zip`` and paste the result
       into ``sha256``. Bump ``DEFAULT_VERSION`` if this is the new latest.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
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
        url="https://zenodo.org/records/20481557/files/DEFault-Bench.zip",
        # Run: shasum -a 256 DEFault-Bench.zip  and paste the result here.
        sha256="",
        license="CC-BY-4.0",
        description=(
            "DEFault++ benchmark v1 — encoder_dataset.csv (3,196 rows) "
            "and decoder_dataset.csv (2,360 rows). Each row is one "
            "fine-tuning run with runtime features and mutation-testing "
            "labels (killed, fault_category, fault_subcategory). "
            "DOI: 10.5281/zenodo.20481557"
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
                extract_root = dest_root / Path(mname).parts[0]
        tar.extractall(dest_root)
    if extract_root is None:
        raise DownloadError(f"empty tarball {tarball}")
    return extract_root


def _extract_zip(archive: Path, dest_root: Path) -> Path:
    """Safely extract ``archive`` (zip) into ``dest_root``, return extract root.

    Refuses any member with an absolute path or ``..`` segment.
    Returns the top-level directory inside the zip, or ``dest_root``
    directly if the zip has no common top-level folder.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise DownloadError(
                    f"refusing to extract unsafe path: {name!r}"
                )
        zf.extractall(dest_root)
        names = zf.namelist()

    # Determine the extract root: common top-level directory, if any.
    top_dirs = {Path(n).parts[0] for n in names if Path(n).parts}
    if len(top_dirs) == 1:
        candidate = dest_root / next(iter(top_dirs))
        if candidate.is_dir():
            return candidate
    return dest_root


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
    archive_name = url.split("/")[-1]
    archive = root / archive_name
    is_zip = archive_name.endswith(".zip")

    if force and root.exists():
        shutil.rmtree(root)

    # Short-circuit: cached extract already present.
    if not force and root.is_dir():
        candidate = _existing_extract_root(root)
        if candidate is not None:
            if not is_zip and verify_manifest:
                try:
                    _verify_manifest(candidate)
                except DownloadError:
                    shutil.rmtree(candidate, ignore_errors=True)
                else:
                    return candidate
            else:
                return candidate

    # If the archive is already on disk and hashes correctly, skip the network.
    if archive.exists() and expected_sha:
        if _sha256_file(archive) != expected_sha:
            archive.unlink()

    if not archive.exists():
        _download_to(url, archive, progress=progress)

    if expected_sha:
        got = _sha256_file(archive)
        if got != expected_sha:
            archive.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 mismatch on download from {url}: "
                f"expected {expected_sha[:12]}…, got {got[:12]}…"
            )

    # Clear any pre-existing extract under ``root`` (besides the archive).
    for child in root.iterdir():
        if child == archive:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if is_zip:
        extract_root = _extract_zip(archive, root)
    else:
        extract_root = _extract_tarball(archive, root)
        if verify_manifest:
            _verify_manifest(extract_root)
    return extract_root
