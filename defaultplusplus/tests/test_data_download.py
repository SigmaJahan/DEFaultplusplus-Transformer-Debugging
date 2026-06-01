"""Tests for ``defaultplusplus.data.download_bench`` and the CLI.

Network is never touched — tests build a tiny tarball under tmp_path,
serve it via a ``file://`` URL, and assert the verify + extract logic.
"""
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _build_test_bundle(tmp_path: Path) -> tuple[Path, str]:
    """Create a small tarball that mimics the real bundle layout.

    Returns (tarball_path, sha256_hex).
    """
    bundle_dir = tmp_path / "src" / "defaultpp-bench-test"
    bundle_dir.mkdir(parents=True)

    files = {
        "encoder_merged.csv": "instance_id,is_faulty\nfoo,1\n",
        "decoder_merged.csv": "instance_id,is_faulty\nbar,0\n",
        "README.md": "test bundle",
    }
    for rel, content in files.items():
        (bundle_dir / rel).write_text(content, encoding="utf-8")

    # Build MANIFEST.sha256 for everything we just wrote
    manifest_lines = []
    for rel in sorted(files):
        path = bundle_dir / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {rel}")
    (bundle_dir / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8",
    )

    # Tar it up
    tarball = tmp_path / "bundle.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    return tarball, digest


# ─────────────────────────────────────────────────────────────────────────
# Cache layout
# ─────────────────────────────────────────────────────────────────────────
def test_bench_dir_uses_cache_root_for_known_version(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path))
    from defaultplusplus.data import bench_dir
    p = bench_dir("v1")
    assert p.is_relative_to(tmp_path)
    assert p.name == "v1"
    assert p.parent.name == "bench"


# ─────────────────────────────────────────────────────────────────────────
# Version registry guard rails
# ─────────────────────────────────────────────────────────────────────────
def test_unknown_version_raises():
    from defaultplusplus.data import download_bench
    with pytest.raises(KeyError, match="unknown bench version"):
        download_bench(version="v999")


def test_unpublished_version_raises_typed_error(monkeypatch, tmp_path):
    """A version with url=None must give callers an actionable error.

    Stub a pretend-future-version with no upload URL into the registry
    so the published v1 record (https://doi.org/10.5281/zenodo.20481557)
    keeps working in the rest of the suite.
    """
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path))
    from defaultplusplus.data import (
        BENCH_VERSIONS, BenchVersion, DatasetNotPublishedError,
        download_bench,
    )
    monkeypatch.setitem(BENCH_VERSIONS, "v_unpublished", BenchVersion(
        name="v_unpublished",
        url=None,
        sha256="0" * 64,
        license="CC-BY-4.0",
        description="placeholder for the typed-error test",
    ))
    with pytest.raises(DatasetNotPublishedError) as exc_info:
        download_bench(version="v_unpublished")
    msg = str(exc_info.value)
    assert "has not been uploaded" in msg
    assert "url_override" in msg


# ─────────────────────────────────────────────────────────────────────────
# Full download → verify → extract round trip via file:// URL
# ─────────────────────────────────────────────────────────────────────────
def test_download_with_url_override_extracts_and_verifies(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import download_bench

    tarball, digest = _build_test_bundle(tmp_path)
    url = tarball.as_uri()  # file:///tmp/.../bundle.tar.gz

    extract = download_bench(
        version="v1",
        url_override=url,
        sha256_override=digest,
        progress=False,
    )
    assert extract.is_dir()
    # The bundle README and CSVs should be present.
    assert (extract / "encoder_merged.csv").read_text().startswith("instance_id")
    assert (extract / "MANIFEST.sha256").exists()


def test_download_short_circuits_when_cache_is_intact(monkeypatch, tmp_path):
    """Second call with same URL should NOT re-extract (cache hit)."""
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import download_bench

    tarball, digest = _build_test_bundle(tmp_path)
    url = tarball.as_uri()

    first = download_bench(version="v1", url_override=url,
                           sha256_override=digest, progress=False)
    # Touch a sentinel inside the extract; if the second call wipes it,
    # we know the short-circuit didn't fire.
    sentinel = first / "_cache_sentinel"
    sentinel.write_text("kept")

    second = download_bench(version="v1", url_override=url,
                            sha256_override=digest, progress=False)
    assert second == first
    assert sentinel.exists(), (
        "second call re-extracted; cache short-circuit didn't fire"
    )


def test_force_redownload_wipes_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import download_bench

    tarball, digest = _build_test_bundle(tmp_path)
    url = tarball.as_uri()

    extract = download_bench(version="v1", url_override=url,
                             sha256_override=digest, progress=False)
    sentinel = extract / "_cache_sentinel"
    sentinel.write_text("kept")

    second = download_bench(version="v1", url_override=url,
                            sha256_override=digest, force=True,
                            progress=False)
    assert second == extract
    assert not sentinel.exists(), "force=True did not wipe the extract"


# ─────────────────────────────────────────────────────────────────────────
# Integrity failures
# ─────────────────────────────────────────────────────────────────────────
def test_sha256_mismatch_raises_and_drops_partial(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import DownloadError, download_bench

    tarball, _ = _build_test_bundle(tmp_path)
    url = tarball.as_uri()
    bogus_sha = "0" * 64

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        download_bench(version="v1", url_override=url,
                       sha256_override=bogus_sha, progress=False)

    # The bad tarball must be gone so a retry doesn't keep matching it
    cached = tmp_path / "cache" / "defaultplusplus" / "bench" / "v1" \
        / "defaultpp-bench-v1.tar.gz"
    assert not cached.exists()


def test_corrupt_manifest_fails_verification(monkeypatch, tmp_path):
    """If a file inside the bundle is altered post-extraction, the
    next ``download_bench`` call must detect it and re-fetch."""
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import download_bench

    tarball, digest = _build_test_bundle(tmp_path)
    url = tarball.as_uri()

    extract = download_bench(version="v1", url_override=url,
                             sha256_override=digest, progress=False)

    # Corrupt one extracted file
    (extract / "encoder_merged.csv").write_text("TAMPERED", encoding="utf-8")

    # Next call should detect and re-extract from the still-good tarball
    again = download_bench(version="v1", url_override=url,
                           sha256_override=digest, progress=False)
    assert again == extract
    assert (extract / "encoder_merged.csv").read_text().startswith("instance_id")


def test_unsafe_tar_member_rejected(monkeypatch, tmp_path):
    """Tarballs with absolute paths or ``..`` segments must be rejected."""
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data import DownloadError, download_bench

    # Hand-build a tarball with one safe and one malicious member
    tarball = tmp_path / "evil.tar.gz"
    payload = b"x"
    info = tarfile.TarInfo(name="../escape.txt")
    info.size = len(payload)
    with tarfile.open(tarball, "w:gz") as tar:
        safe = tarfile.TarInfo(name="bundle/file.txt")
        safe.size = len(payload)
        tar.addfile(safe, io.BytesIO(payload))
        tar.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()

    with pytest.raises(DownloadError, match="unsafe path"):
        download_bench(version="v1", url_override=tarball.as_uri(),
                       sha256_override=digest, progress=False)


# ─────────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────────
def test_cli_list_prints_known_versions(capsys):
    from defaultplusplus.data.cli import main
    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "v1" in out


def test_cli_url_without_sha256_errors(capsys):
    from defaultplusplus.data.cli import main
    rc = main(["--url", "file:///tmp/foo.tar.gz"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--url and --sha256" in err


def test_cli_download_with_overrides_succeeds(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DEFAULTPP_CACHE_DIR", str(tmp_path / "cache"))
    from defaultplusplus.data.cli import main

    tarball, digest = _build_test_bundle(tmp_path)
    rc = main([
        "--url", tarball.as_uri(),
        "--sha256", digest,
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ready at" in out
