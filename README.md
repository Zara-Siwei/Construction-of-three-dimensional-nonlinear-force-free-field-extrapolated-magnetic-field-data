# getmag_and_plot

Magnetic Field Extrapolation and 3D Visualization Tool Based on HMI SHARP CEA Data

---

## Overview

getmag_and_plot.py is a Python program for performing:

* Linear Force-Free Field (LFFF) extrapolation
* Nonlinear Force-Free Field (NLFFF) extrapolation using the optimization method
* 3D magnetic field line visualization colored by |J|/|B|

The lower boundary condition is provided by SDO/HMI SHARP CEA vector magnetogram data.

This tool is designed for solar magnetic field research and educational purposes.

---

## Project Structure

project/

├── getmag_and_plot.py
├── hmi_data/                # Input data directory
│   ├── *.Bp.fits
│   ├── *.Bt.fits
│   └── *.Br.fits
│
├── output/                  # Output directory (auto-created)
│   ├── *linearfff.fits
│   ├── *nonlinearfff_guass.fits
│
└── README.md

---

## Input Data

Input files must be HMI SHARP CEA 720s vector magnetograms:

* *.Bp.fits  (phi component)
* *.Bt.fits  (theta component)
* *.Br.fits  (radial component)

Place these files inside the hmi_data/ directory.

---

## Output

Results are saved in the output/ directory:

* Linear force-free extrapolation result (*linearfff.fits)
* Nonlinear force-free extrapolation result (*nonlinearfff_guass.fits)
* 3D magnetic field visualization (if enabled)

---

## Computational Domain

* The x–y plane corresponds to the original HMI SHARP CEA patch region.
* The extrapolation volume extends upward along the z-direction by a height equal to the shorter side of the patch.

The vertical grid size is defined as:

vsize = min(rsize, tsize)

This produces an approximately cubic computational domain.

---

## Field-Line Coloring

Magnetic field lines are colored according to:

|J| / |B|

where:

* J = curl(B)
* |B| is the magnetic field magnitude

Higher values of |J|/|B| indicate stronger electric current concentration.

Default colormap: inferno.

---

## Downsampling

Uniform spatial downsampling is applied before extrapolation:

downsampling_factor = 10

This reduces computational cost while preserving large-scale structure.

You may adjust this parameter inside the script to balance:

* Computational performance
* Spatial resolution

---

## Method

Linear Force-Free Field (LFFF):

* Constant-alpha approximation
* Green’s function–based solution

Nonlinear Force-Free Field (NLFFF):

* Optimization method
* Volume integral minimization of:

L = ∫ |B|² |ω|² dV

where

ω = [ (curl B) × B − (div B) B ] / |B|²

Finite-difference derivatives are used for spatial operators.

---

## How to Run

Run:

python getmag_and_plot.py

Control behavior inside the script:

save_lfff = True
save_nlfff = True
if_plot = True

---

## Example Output

Below is an example of 3D magnetic field lines colored by |J|/|B|:

![Example Field Lines](docs/example_plot.png)

Replace this image with your own visualization output.

---

## Dependencies

* numpy
* matplotlib
* astropy
* opencv-python
* tqdm

Install via:

pip install numpy matplotlib astropy opencv-python tqdm

---

## Notes

* Designed for research and educational use.
* NLFFF convergence may vary depending on active region complexity.
* Performance depends strongly on grid resolution and downsampling factor.
* Boundary weighting function wf can be modified (e.g., cosine tapering) to reduce edge effects.

---

## Author

Zara
Solar Magnetic Field Extrapolation Research

---
