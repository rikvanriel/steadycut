"""Model weights this project needs, fetched lazily and pinned exactly.

Nothing here downloads at import, at install, or during the test suite: every
entry resolves through `ensure()`, which checks the local cache (sha256 of
file CONTENT, read off downloaded bytes -- never off Hub cache filenames,
which use LFS-pointer naming on some files) and fetches only a missing or
corrupt file, from a pinned revision. Offline, or on a hash
mismatch, it raises naming the feature, the manual URL, and the
`STEADYCUT_MODELS_DIR` override -- never a bare connection error, never a
half-written file (atomic rename), never weights in git.

Adding a model: one MODELS entry (repo, rev, files with content shas from a
verified local copy), the feature that calls `ensure()`, and the dep that
runs it in the `ml` extra. No new mechanism per model.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    repo: str        # "owner/name" on the Hub, or "local" for user-hosted
    rev: str         # full commit sha the bytes were verified against
    path: str        # repo-relative path of the file
    sha256: str      # of the file CONTENT (what a resolve-download yields)
    size: int        # bytes, to fail fast on truncation


@dataclass(frozen=True)
class Model:
    name: str
    needed_by: str   # the feature that calls ensure(), for error messages
    files: tuple[ModelFile, ...]


# Pinned 2026-10-03 from verified local copies (content shas read off disk,
# revisions from the Hub API). The depth ONNX is the exact bytes validated in
# the Task-8 probes, hosted by the project until then (see needed_by).
MODELS: dict[str, Model] = {
    "segformer-b2-ade": Model(
        name="segformer-b2-ade",
        needed_by="assist-mode pseudo-labelling (steadycut.framing.assist)",
        files=(
            ModelFile("nvidia/segformer-b2-finetuned-ade-512-512",
                      "de01bae28967510f9ddd496c60a969357195400c",
                      "config.json",
                      "ee7400840fdb1e5045f0b2eba78bf053df8e33a309c4acec31705a48c8cc5c00",
                      6885),
            ModelFile("nvidia/segformer-b2-finetuned-ade-512-512",
                      "de01bae28967510f9ddd496c60a969357195400c",
                      "preprocessor_config.json",
                      "8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c",
                      271),
            ModelFile("nvidia/segformer-b2-finetuned-ade-512-512",
                      "de01bae28967510f9ddd496c60a969357195400c",
                      "pytorch_model.bin",
                      "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3",
                      110000905),
        ),
    ),
    "dinov2-small": Model(
        name="dinov2-small",
        needed_by="assist-mode triage (steadycut.framing.assist)",
        files=(
            ModelFile("facebook/dinov2-small",
                      "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                      "config.json",
                      "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
                      547),
            ModelFile("facebook/dinov2-small",
                      "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                      "preprocessor_config.json",
                      "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
                      436),
            ModelFile("facebook/dinov2-small",
                      "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                      "model.safetensors",
                      "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
                      88249960),
        ),
    ),
    "depth-anything-v2-small-onnx": Model(
        name="depth-anything-v2-small-onnx",
        needed_by="assist-mode triage (steadycut.framing.assist)",
        files=(
            # Provenance recovered 2026-10-03 by content hash: the validated
            # bytes are onnx-community's export (producer tag in the graph
            # says huggingface-optimum style, input pixel_values). The two
            # lookalikes checked (Xenova V1-small, HF transformers V2-small)
            # are different models, not different packagings.
            ModelFile("onnx-community/depth-anything-v2-small-ONNX",
                      "c3b67641fd837b2368757101311e5d21e511441e",
                      "onnx/model_quantized.onnx",
                      "ed7680047cd210143f9f9c4468d049c1f19f8a7a58631d52868831376fe152c9",
                      162005),
            ModelFile("onnx-community/depth-anything-v2-small-ONNX",
                      "c3b67641fd837b2368757101311e5d21e511441e",
                      "onnx/model_quantized.onnx_data",
                      "1595c419ac7d75a356c6835890f2e64fd370a38444ddfcb741ca6b6b96e1a42a",
                      38414848),
        ),
    ),
}


class ModelUnavailable(Exception):
    """A needed model is absent and cannot be fetched (offline, or corrupt)."""


def models_dir() -> Path:
    """$STEADYCUT_MODELS_DIR, else $XDG_CACHE_HOME/steadycut/models."""
    override = os.environ.get("STEADYCUT_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg) / "steadycut" / "models"


def _url(f: ModelFile) -> str:
    if f.repo == "local":
        raise ModelUnavailable(
            f"no remote for {f.path}: copy it to "
            f"{models_dir()}/ first")
    if f.repo.startswith("file://"):
        # Air-gapped setups and the offline test suite: a file transport is
        # real transport, verified the same way.
        return f"{f.repo}/{f.path}"
    if f.rev == "TBD" or f.repo.startswith("TBD"):
        raise ModelUnavailable(
            f"{f.path} has no pinned remote yet (see MODELS entry): upload "
            f"the validated bytes to the project account and pin repo/rev, "
            f"or place them at {models_dir()}/")
    return f"https://huggingface.co/{f.repo}/resolve/{f.rev}/{f.path}"


def _verified(path: Path, f: ModelFile) -> bool:
    if not path.exists() or path.stat().st_size != f.size:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == f.sha256


def ensure(name: str, directory: Path | None = None) -> Path:
    """Local directory holding a verified model, fetching what is missing.

    Returns the directory; files land directly inside it (transformers-style
    repos load from a directory, ONNX pairs sit side by side). Raises
    ModelUnavailable with the manual path when offline or corrupt -- the
    caller (a feature, never import time) turns that into its refusal.
    """
    try:
        model = MODELS[name]
    except KeyError:
        raise ModelUnavailable(f"unknown model {name!r}") from None
    root = Path(directory) if directory else models_dir() / name
    root.mkdir(parents=True, exist_ok=True)
    for f in model.files:
        dst = root / Path(f.path).name
        if _verified(dst, f):
            continue
        url = _url(f)  # raises before any network on unpinned entries
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
                with urllib.request.urlopen(url) as resp:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        tmp.write(chunk)
                tmp_path = Path(tmp.name)
        except Exception as exc:
            raise ModelUnavailable(
                f"could not fetch {f.path} for {model.needed_by}: {exc}; "
                f"fetch {url} manually into {root}/, or set "
                f"STEADYCUT_MODELS_DIR") from exc
        if not _verified(tmp_path, f):
            tmp_path.unlink(missing_ok=True)
            raise ModelUnavailable(
                f"fetched {f.path} failed verification for "
                f"{model.needed_by}; file removed, refusing to use it")
        tmp_path.replace(dst)
    return root
