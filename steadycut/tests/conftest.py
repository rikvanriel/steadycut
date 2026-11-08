"""Shared Qt application for the GUI test modules.

PySide6 allows exactly one QApplication per process: each test module used
to build its own module-scoped singleton, so whichever module ran second
errored at fixture setup ("destroy the QApplication singleton first").
One session-scoped instance serves every module; creation is lazy so
collecting the Qt-free test modules never imports PySide6.
"""
import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    PySide6 = pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
