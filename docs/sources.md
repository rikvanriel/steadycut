# Public sources

Upstream references consulted while building the stitching and tone
paths. Listed so a later reader can check the mechanism against the
original instead of trusting our comments.

- **Lens geometry.** The equidistant `r = f·theta` core of `unproject`
  (and the per-lens radial terms around it) is the Kannala–Brandt
  wide-angle model: Juho Kannala and Sami S. Brandt, "A Generic Camera
  Model and Calibration Method for Conventional, Wide-Angle, and
  Fish-Eye Lenses", IEEE TPAMI 2006.
  https://users.aalto.fi/~kannalj1/publications/tpami2006.pdf
  (DOI: 10.1109/TPAMI.2006.153)

- **Fisheye→equirect projection.** All `v360=input=fisheye` mappings
  (graph and stills) follow ffmpeg's equidistant fisheye model, where
  `ih_fov`/`iv_fov` set the input angular scale — the knob the FOV
  rescale turns. Filter docs, v360 section:
  https://ffmpeg.org/ffmpeg-filters.html
  Implementation (the mapping our FOV numbers feed):
  https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_v360.c

- **Pyramid blending.** `multiband_merge` is a multiresolution spline
  after Burt & Adelson: each band blends over a zone proportional to
  its own wavelength, so coarse tone dissolves while fine detail
  survives the cut. Peter J. Burt and Edward H. Adelson, "A
  Multiresolution Spline With Application to Image Mosaics", ACM TOG
  1983. https://persci.mit.edu/pub_pdfs/spline83.pdf

What we measured ourselves (not from these sources): the effective
lens FOVs (Studio-separation test), the 1.34× inter-lens deflare
ratio, per-window tone fits, and all gates/budgets.
