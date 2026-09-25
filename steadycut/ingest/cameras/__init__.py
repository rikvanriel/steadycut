"""Importing this package registers every camera profile.

Camera definitions live one per file (insta360_x4.py, later insta360_x5.py,
gopro_max.py, ...); each calls register() at import. Anything that resolves a
profile imports this package, so registration is guaranteed before lookup.
"""
from steadycut.ingest.cameras import insta360_x4  # noqa: F401  (registers the X4)
