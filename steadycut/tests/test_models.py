"""Model fetching without a network: the suite must pass offline.

ensure() is the only thing that touches the network in this module, and
these tests never let it: manifests are monkeypatched to file-present,
TBD-remote, or unknown entries, so every path resolves before urlopen.
"""
import hashlib

import pytest

from steadycut import models
from steadycut.models import Model, ModelFile, ModelUnavailable


def _entry(tmp_path, name="tiny.bin", content=b"weights-bytes",
           repo="owner/repo", rev="abc123"):
    p = tmp_path / name
    p.write_bytes(content)
    f = ModelFile(repo=repo, rev=rev, path=name,
                  sha256=hashlib.sha256(content).hexdigest(),
                  size=len(content))
    return f, p


def test_verified_files_return_without_fetching(tmp_path, monkeypatch) -> None:
    d = tmp_path / "m"
    d.mkdir()
    (d / "tiny.bin").write_bytes(b"weights-bytes")
    f = ModelFile(repo="owner/repo", rev="abc123", path="tiny.bin",
                  sha256=hashlib.sha256(b"weights-bytes").hexdigest(),
                  size=len(b"weights-bytes"))
    monkeypatch.setitem(models.MODELS, "t",
                        Model(name="t", needed_by="tests", files=(f,)))
    out = models.ensure("t", directory=d)
    assert (out / "tiny.bin").read_bytes() == b"weights-bytes"


def test_unpinned_remote_refuses_before_network(tmp_path, monkeypatch) -> None:
    f, _ = _entry(tmp_path, repo="TBD-x", rev="TBD")
    monkeypatch.setitem(models.MODELS, "t",
                        Model(name="t", needed_by="tests", files=(f,)))
    with pytest.raises(ModelUnavailable):
        models.ensure("t", directory=tmp_path / "m")


def test_corrupt_file_is_refetched_not_used(tmp_path, monkeypatch) -> None:
    upstream = tmp_path / "up"
    upstream.mkdir()
    (upstream / "tiny.bin").write_bytes(b"weights-bytes")
    f = ModelFile(repo=f"file://{upstream}", rev="test", path="tiny.bin",
                  sha256=hashlib.sha256(b"weights-bytes").hexdigest(),
                  size=len(b"weights-bytes"))
    d = tmp_path / "m"
    d.mkdir()
    (d / "tiny.bin").write_bytes(b"tampered")
    monkeypatch.setitem(models.MODELS, "t",
                        Model(name="t", needed_by="tests", files=(f,)))
    out = models.ensure("t", directory=d)
    assert (out / "tiny.bin").read_bytes() == b"weights-bytes"


def test_unknown_model_is_unavailable(tmp_path) -> None:
    with pytest.raises(ModelUnavailable):
        models.ensure("no-such-model", directory=tmp_path)


def test_models_dir_override_is_respected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STEADYCUT_MODELS_DIR", str(tmp_path / "custom"))
    assert models.models_dir() == tmp_path / "custom"


def test_manifest_entries_carry_verification() -> None:
    for name, model in models.MODELS.items():
        assert model.needed_by, name
        for f in model.files:
            assert len(f.sha256) == 64, (name, f.path)
            assert f.size > 0, (name, f.path)
            assert len(f.rev) >= 3, (name, f.path)
