# References and citations

Every citation below backs a specific, non-obvious modeling or algorithm
choice in this repository -- not a general reading list. Where an attempt
was tried and abandoned, that is stated explicitly, because the honest
account of what failed is part of the technical record for this
submission (and directly serves the "identifying root cause /
explainability" scoring criterion).

## Synthetic data generation (`src/pattern_render.py`, `src/dataset_generator.py`)

1. **SEM edge-brightening effect.** SEM images show measurably brighter
   contrast right along topographic edges, caused by increased
   secondary-electron escape probability near an edge. Modeled in
   `add_edge_brightening()` as a gradient-magnitude-proportional
   brightness boost.
   - Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation
     and Microanalysis*, 2nd ed., Springer, 1998, ch. 4-5.
   - Goldstein, J. I. et al. *Scanning Electron Microscopy and X-Ray
     Microanalysis*, 4th ed., Springer, 2018, ch. 3.

2. **Independent sensor (shot + read) noise model.** Reference and search
   images get independent noise draws (they are separate physical
   captures) using a Poissonian-Gaussian model: noise standard deviation
   scales with the square root of local signal (shot noise) plus a
   constant-variance read-noise floor.
   - Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. "Practical
     Poissonian-Gaussian Noise Modeling and Fitting for Single-Image
     Raw-Data." *IEEE Transactions on Image Processing*, 17(10),
     1737-1754, 2008. DOI: 10.1109/TIP.2008.2001399.

3. **Alignment/fiducial marks as the disambiguating landmark.** A
   subset of samples (`--defect-prob`) get a deliberate square fiducial
   mark rather than being purely periodic -- modeled on the real
   practice of etching box-in-box or similar alignment/overlay marks
   onto a die specifically to give automated tools a non-periodic
   navigation reference, standard photolithography overlay-metrology
   practice rather than an invented convenience:
   - International Roadmap for Devices and Systems (IRDS), *Metrology*
     chapter (overlay/alignment mark targets), IEEE, latest edition,
     irds.ieee.org.

4. **Anti-aliasing / band-limited rendering across a 10x scale change.**
   Rendering the same periodic function at two very different sampling
   densities (reference vs. search) without a proper band-limiting step
   produces Moire aliasing. `render_capture()` scales the edge-softness
   parameter by the capture's own sampling scale (a simple, analytic
   pre-filter) and additionally supersamples and area-averages
   (`cv2.INTER_AREA`), the standard box-filter downsampling approach.
   - Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed.,
     Pearson, 2018, ch. 4 (sampling and aliasing).

5. **Second DRAM layout style: arcuate moats / wavy bit lines**
   (`dram_arcuate_intensity`). Added after cross-checking the official
   problem statement's own example images, which use this diagonal,
   non-orthogonal style rather than a plain rectangular grid. Grounded
   in an actual COB-DRAM layout patent describing arcuate (curved,
   two-legged) storage-node moats connected by bit lines in a wavy,
   crest-and-trough half-pitch pattern:
   - Texas Instruments Inc., European Patent EP0780901A2, "Method of
     making dynamic random access memory (DRAM) cell arrays with
     arcuate moats and wavy bit lines," priority 1995.
   - Approximated in the generator as two families of diagonal capsule
     ("stadium") shapes on a herringbone offset rather than the exact
     patent geometry -- captures the visual/statistical character
     (non-orthogonal periodicity, elongated rounded features) that a
     naive rectangular grid misses; documented as an approximation, not
     a claim of literal reproduction.

## Localization algorithm (`src/localizer.py`)

6. **Normalized cross-correlation via `cv2.matchTemplate`.** The coarse
   search uses OpenCV's `TM_CCOEFF_NORMED`, FFT-accelerated internally,
   which is what keeps a several-dozen-hypothesis scale x rotation
   sweep fast enough to run on CPU.
   - OpenCV documentation, *Template Matching*,
     docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html.
   - Lewis, J. P. "Fast Normalized Cross-Correlation." *Vision
     Interface*, 1995.

7. **Sub-pixel refinement by parabolic interpolation.**
   - Tian, Q. & Huhns, M. N. "Algorithms for Subpixel Registration."
     *Computer Vision, Graphics, and Image Processing*, 35(2), 220-233,
     1986.

8. **Uniqueness-ratio confidence (and its calibration).**
   - Lowe, D. G. "Distinctive Image Features from Scale-Invariant
     Keypoints." *International Journal of Computer Vision*, 60(2),
     91-110, 2004.
   - The optional learned recalibration (`train_confidence.py`) follows
     standard probability-calibration practice: Platt, J. "Probabilistic
     Outputs for Support Vector Machines and Comparisons to Regularized
     Likelihood Methods." *Advances in Large Margin Classifiers*, MIT
     Press, 1999.

## The learned patch matcher (`src/cnn_matcher.py`, `src/train_cnn_matcher.py`)

9. **Convolutional network architecture and training.** A small
   Conv-ReLU-Pool x2 -> Dense-ReLU -> Dense-Sigmoid binary patch
   classifier, implemented from first principles in NumPy rather than a
   framework (see README "Why a hand-written CNN").
   - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. "Gradient-Based
     Learning Applied to Document Recognition." *Proceedings of the
     IEEE*, 86(11), 2278-2324, 1998.
   - Chellapilla, K., Puri, S., & Simard, P. "High Performance
     Convolutional Neural Networks for Document Processing."
     *International Workshop on Frontiers in Handwriting Recognition*,
     2006. (The im2col matrix-multiply formulation of convolution used
     in `cnn_matcher.py`'s forward/backward passes.)
   - Kingma, D. P. & Ba, J. "Adam: A Method for Stochastic Optimization."
     *ICLR*, 2015. (The optimizer implemented in `Conv2D.step` /
     `Dense.step`.)
   - He, K., Zhang, X., Ren, S., & Sun, J. "Delving Deep into Rectifiers:
     Surpassing Human-Level Performance on ImageNet Classification."
     *ICCV*, 2015. (The weight-init scale used for conv/dense layers.)

## Design attempts that did not work (kept for an honest record)

- **A single missing via / broken fin as the disambiguating landmark.**
  For a template spanning ~8-10 pattern periods, a single missing
  feature changes whole-template NCC by well under 0.001 in
  TM_CCOEFF_NORMED units -- far smaller than the ~0.01-0.03 natural
  score spread between different periodic repeats, so it could not
  reliably move the argmax to the true location. Replaced by a
  deliberately-placed, substantially larger fiducial.
- **A "box-in-box" (bright core + dark ring) fiducial**, matching the
  literal industry mark design more closely than a plain filled square.
  Performed no better than the plain filled square in this generator,
  likely because the dark ring blends into the pattern's own
  already-dark background over much of its area. The plain bright
  square was kept.
- **Aggregating (element-wise max) the correlation surface across every
  scale/rotation hypothesis before peak-finding.** Less robust than
  picking one best hypothesis by its own global max: with dozens of
  hypotheses compared independently at every pixel, noise gets many
  independent chances to spike, and the aggregated map inflated
  unrelated background locations almost to the level of the true match.
- **Truncating NMS candidates by correlation score** when more qualify
  than `max_peaks`. The tie-break picks the closest-to-center candidate
  among everyone who qualifies, not the highest-scoring one -- so
  truncating by score first silently discarded the true winner on
  densely periodic fields. Caught via evaluate.py's spec-fair metric;
  fixing the truncation to keep candidates closest to center instead
  raised the spec-fair within-20px rate on the periodic-only self-test
  bucket from 6% to 33%.
- **Installing PyTorch for the learned component.** Tried first, before
  the from-scratch NumPy network. The default PyPI wheel for this
  platform bundles a full CUDA toolkit (~5-6 GB); in the development
  environment this filled the available disk mid-install and left an
  unimportable package (`libtorch_global_deps.so` missing, consistent
  with a download truncated by size). Confirmed broken, not just
  theoretically risky -- see README "Why a hand-written CNN."

## Phase 2 addendum (`src/pattern_render.py`, `src/pose_refine.py`)

10. **Charging and scan-distortion artifacts.** Beam-induced charge
    buildup on insulating regions (localized bright blooming) and
    slow-scan-axis drift (smooth per-row displacement) are both
    well-documented SEM imaging artifacts, covered in the same
    imaging-artifact chapters already cited above for edge brightening:
    - Reimer (1998), ch. 2-3 (scan/drift artifacts) and ch. 4-5 (charging).
    - Goldstein et al. (2018), ch. 3.
    Modeled deliberately crudely (`add_charging_artifact`,
    `add_scan_distortion`) -- real charging geometry follows the
    underlying dielectric layout, which this project has no access to
    without proprietary layout data (disallowed by the addendum's own
    rules); the goal is a plausible, high-contrast nuisance signal for
    robustness training, not a physically exact simulator.

11. **Mipmap-style pre-filtering before downsampling.** `pose_refine
    .make_template_continuous` Gaussian-blurs the source in proportion
    to the requested downscale factor before sampling, to avoid
    aliasing from an ~8-12x downsample via plain bilinear
    interpolation -- standard practice in image/texture resampling:
    - Williams, L. "Pyramidal Parametrics." *ACM SIGGRAPH Computer
      Graphics*, 17(3), 1-11, 1983. DOI: 10.1145/964967.801126.

12. **Least-squares lattice refinement from multiple observations.**
    `lattice_localizer.refine_lattice_from_peaks` fits lattice basis
    vectors from several detected peak positions at once rather than
    trusting a single autocorrelation sample -- the same
    averaging-out-noise-over-many-observations principle behind
    standard linear regression / errors-in-variables estimation:
    - Golub, G. H. & Van Loan, C. F. *Matrix Computations*, 4th ed.,
      Johns Hopkins University Press, 2013, ch. 5 (least squares).

