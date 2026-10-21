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

- **Panorama gain compensation.** `measure_gain` matches overlap
  means between the lenses with a prior toward unity — the Brown &
  Lowe gain formulation: Matthew Brown and David G. Lowe, "Automatic
  Panoramic Image Stitching using Invariant Features", IJCV 2007
  (Sections 6–7: gain compensation, multi-band blending).
  https://www.cs.ubc.ca/~lowe/papers/07brown.pdf

- **Scale-invariant features.** SIFT matching (`cv2.SIFT_create` in
  corridor, detection, repair) is Lowe's detector/descriptor: David
  G. Lowe, "Distinctive Image Features from Scale-Invariant
  Keypoints", IJCV 2004.
  https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

- **Robust estimation.** RANSAC consensus for geometric fits
  (essential/affine; `far_track`, correspondence code) is Fischler &
  Bolles: Martin A. Fischler and Robert C. Bolles, "Random Sample
  Consensus: A Paradigm for Model Fitting with Applications to Image
  Analysis and Automated Cartography", CACM 1981.
  https://www.sri.com/wp-content/uploads/2021/12/ransac-publication.pdf

- **Dense optical flow.** `corridor_flow` disparity uses OpenCV's DIS
  (Dense Inverse Search): Till Kroeger et al., "Fast Optical Flow
  using Dense Inverse Search", ETH CVL 2016 (arXiv:1603.03590).
  https://arxiv.org/pdf/1603.03590v1.pdf

- **IMU orientation filtering.** `attitude_vqf` wraps the VQF
  quaternion gyro/accel filter: Daniel Laidig and Thomas Seel, "VQF:
  Highly accurate IMU orientation estimation with bias estimation and
  magnetic disturbance rejection", Information Fusion 2023.
  https://doi.org/10.1016/j.inffus.2022.10.014
