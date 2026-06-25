# NLFFF Demo

This folder is a clean demo pipeline for an HMI SHARP CEA active region:

- download `Bp`, `Bt`, `Br` from JSOC;
- crop, downsample, smooth, and flux-balance the boundary;
- compute a potential field by FFT;
- relax an optimization-style NLFFF field from the observed bottom vector boundary;
- save both fields as FITS;
- render field lines with PyVista, not Matplotlib.

Default target:

- NOAA AR: `11158`
- HARP: `377`
- Time: `2011.02.15_00:00:00_TAI`
- Why this one: AR 11158 is a classic sigmoid / strong-PIL / X2.2-event active region used heavily in NLFFF studies, and it is much better for a twisted-field demo than quiet or diffuse regions.

Useful references:

- Zhao et al. 2014, "Temporal Evolution of the Magnetic Topology of the NOAA Active Region 11158": https://arxiv.org/abs/1404.5004
- Bobra et al. 2014, "The HMI Vector Magnetic Field Pipeline: SHARPs": https://arxiv.org/abs/1404.1879
- Wiegelmann 2008, "Optimization code with weighting function for the reconstruction of coronal magnetic fields": https://arxiv.org/abs/0802.0124

DRMS check used:

```text
hmi.Mharp_720s[377][2011.02.15_00:00:00_TAI]
HARPNUM=377, NOAA_ARS=11158
```

## Run

Use your known-good Python:

```powershell
& "G:\python_projects\envs\WPy64-31241\python-3.12.4.amd64\python.exe" "E:\python项目\NLFFF-master\Perfect-NLFFF-master\perfect_nlfff_demo.py"
```

The main parameters are at the top of `perfect_nlfff_demo.py`.

Good first-run settings are already selected:

```python
MAX_XY_POINTS = 120
Z_SIZE = 72
NLFFF_ITERATIONS = 450
WEIGHT_TAPER_CELLS = 10
```

For a better-looking but heavier run, try:

```python
MAX_XY_POINTS = 180
Z_SIZE = 96
NLFFF_ITERATIONS = 2000
FIELDLINE_SEEDS = 350
```

Raise only one or two parameters at a time. Memory and runtime scale quickly with the 3D grid size.

Double-click / interactive PyVista use is enabled by default:

```python
PYVISTA_OFF_SCREEN = False
SHOW_INTERACTIVE_WINDOWS = True
INTERACTIVE_VIEW_TARGETS = ("nlfff",)
```

With these settings the script still saves PNG files, then opens an interactive PyVista window for the NLFFF result. Change `INTERACTIVE_VIEW_TARGETS` to `("potential", "nlfff")` if you also want the potential-field window.

## Field-Line Style

The field-line controls are all near the top of `perfect_nlfff_demo.py`.

Useful seed modes:

- `pil_shear`: high transverse field near the polarity inversion line. Best first choice for flux-rope-like low-lying twisted lines.
- `strong_bz`: strongest vertical-field footpoints. Good for full active-region connectivity, but often less focused on the rope.
- `strong_bh`: strongest horizontal-field footpoints. Good for sheared core fields.
- `mixed_bz_bh`: strong vertical and transverse field together.
- `polarity_balanced`: equal positive/negative polarity footpoints.
- `grid`: uniform seeds; useful for debugging.
- `j_over_b_low`: low-corona current proxy; often highlights current-carrying bundles.
- `random_weighted`: repeatable random seeds weighted toward the shear channel.

Useful color modes:

- `j_over_b`: relative current strength, good for current channels and reconnection-looking structures.
- `bmag`: field strength.
- `height`: emphasizes loop height and geometry.
- `alpha`: signed `J.B/B^2`, a force-free-alpha/twist proxy.
- `current_helicity`: signed `J.B`, useful for helicity/current handedness.
- `solid`: one clean color, no colorbar; useful when geometry matters more than diagnostics.

Good display combinations:

- Pretty flux-rope demo: `FIELDLINE_SEED_MODE = "pil_shear"`, `FIELDLINE_COLOR_MODE = "j_over_b"`, `FIELDLINE_CMAP = "turbo"`.
- Twist sign check: `FIELDLINE_SEED_MODE = "pil_shear"`, `FIELDLINE_COLOR_MODE = "alpha"`, `FIELDLINE_SIGNED_CMAP = "coolwarm"`.
- Clean publication-style geometry: `FIELDLINE_COLOR_MODE = "solid"` and choose `FIELDLINE_SOLID_COLOR`.

## Outputs

The default output folder is:

```text
Perfect-NLFFF-master/output/harp377_20110215_000000
```

Files:

- `potential_field.fits`
- `nlfff_field.fits`
- `diagnostics.csv`
- `potential_fieldlines.png`
- `nlfff_fieldlines.png`

FITS axis order:

```text
component,z,y,x
component 0 = Bx = Bp
component 1 = By = -Bt
component 2 = Bz = Br
unit = Gauss
```

HTML export is attempted, but PyVista needs extra `trame` dependencies for that. PNG output works without them.

## Important Caveats

This is a demo-grade NLFFF implementation, not a production replacement for mature codes such as the Wiegelmann optimization code or magnetofrictional codes used in many papers.

Reasons:

- HMI photospheric vector fields are not force-free. NLFFF codes usually preprocess the bottom boundary.
- Real published flux-rope figures often use carefully chosen seed points, rendering, preprocessing, and sometimes data-constrained flux-rope insertion or magnetofrictional evolution.
- A visually beautiful twisted flux rope is not guaranteed from a single raw vector magnetogram.

## Hard Problems In The Old Script

The old scripts are useful as a translation experiment, but several details can make the extrapolation look rough or physically weak:

- The saved LFFF file is normalized by `bzmax`, while the NLFFF file is rescaled to Gauss. That makes direct comparisons misleading.
- `wf` is effectively all ones. Optimization NLFFF needs side/top boundary tapering; otherwise fixed side/top errors pollute the volume.
- There is no serious photospheric preprocessing. Raw SHARP data often violate force-free consistency.
- The Green-function potential/LFFF loop is very expensive and numerically fragile near the singular point.
- The code executes work at import time and mixes download, extrapolation, diagnostics, and plotting in one global script.
- Matplotlib 3D line rendering can hide or reorder layers, which can make good field lines look bad.
- The field-line seeds are not tuned to the polarity inversion line or high-shear channel.
- Diagnostics are not written, so it is hard to know whether the optimization is actually improving.

The new script fixes the workflow issues and adds diagnostics, but the physical limitations above still matter.
