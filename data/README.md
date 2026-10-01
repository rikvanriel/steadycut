# Framing labels

`framing-labels.jsonl` — one JSON object per line — is the human judgement on how
MTB helmet footage from the 360 should be framed. It exists because framing
criteria are easy to build and easy to be wrong about, and a criterion that was
never checked against a person looking at frames is a criterion that has only
agreed with itself.

## What is in a row

| field | meaning |
|---|---|
| `recording` | which `.insv` segment the frame comes from |
| `seconds` | where in that recording |
| `framing_class` | `good`, `low`, `bad`, `not-riding`, `no-horizon`, `on-target`, `position-given`, `unlabelled` |
| `boundary_position_pct` | the rider's reading of the 10–90 grid, or `null` |
| `labeller` | **`rider`**, **`vlm`** (a model, uncorrected), or `none` |
| `note` | the rider's own words where they gave any |
| `sheet` | which sampling round produced the row |
| `conflict` / `also_labelled_at` | present when a frame got two answers |

`labeller` is the field that matters most and the one most easily lost. A frame
the rider called "perfect" and a frame a model called "good" are different
kinds of evidence, and a dataset that cannot tell them apart cannot be used to
measure whether a model is any good. The `vlm` rows are kept *uncorrected*
alongside the rider's corrections to two of them, because the agreement between
the two is itself the measurement: 22 of 24 on the down-edge round.

## Known problems with this data, stated rather than buried

- **92 rows is far short of what a correlation needs.** A Spearman ρ of 0.2 wants
  about 194 samples at 80% power; below 100, small-sample error inflates apparent
  effects. Treat any correlation computed on this file as indicative only.
- **It is six segments of one rider on one day.** No season, no other body, no
  third-person material. Anything it concludes is a conclusion about this corpus.
- **The frames are not tracked.** They are ~100 rendered mp4s and they rebuild
  from the `.insv` plus the recipe named in each row's `sheet`, which is the same
  trade as everywhere else here: track the labels, rebuild the pixels.
- **Two rows carry a recorded conflict** rather than a silent resolution.
- **Positions come from a downscaled contact sheet.** The rider read them off
  tiles, so they carry the error of that scale. The rider's own position reads
  were more precise than a model's on the same tiles, but they are not click-
  accurate and should not be treated as such.

## What it was collected for

The framing target was assumed constant and is not: the rider's positions have a
median of 30 and a standard deviation of 13.6 points of frame height, which is
6.8× the ground cue's held-out error. But the cue reads to a mean of 1.9 points
on frames the rider called already-on-target, and its median matches the constant
the tool already ships — so the target's variance was never evidence that the
instrument was wrong. The remaining gap is a **visible-boundary** detector, and
this file is the labelled set such a detector would be measured against.
