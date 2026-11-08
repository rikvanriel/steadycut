# Contributing (standing rules)

## Every UI element runs in CI

`steadycut/gui/widgets.py` must stay at least 90% line-covered by the
test suite (`test_gui_widgets.py`, `test_proxy_gui.py`, ...). CI fails
below that. In practice this means: **every widget, handler, and custom
paint path you add or touch comes with a test that instantiates it and
drives it** (QTest clicks/drags for mouse handlers, direct calls plus
repaint for paint code). The rule exists because the timeline click bug
shipped: the handler was never executed anywhere but a human hand.

New app-level flows (render, cancel, copy, refusals) get interaction
tests through `Main` the same way.
