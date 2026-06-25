from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import drms
import numpy as np
import pyvista as pv
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from tqdm import trange


# =============================================================================
# Default click-run demo settings: edit here first
# =============================================================================
#
# 默认直接运行本脚本时，会做下面这件事：
#   1. 使用 NOAA AR 11158 / HARP 377 / 2011-02-15 这个经典活动区；
#   2. 读取或自动下载 SHARP CEA 的 Bp/Bt/Br；
#   3. 输出势场和 NLFFF 的 FITS；
#   4. 用 PyVista 画浅色背景、右侧竖 colorbar 的磁力线图；
#   5. 最后弹出 NLFFF 交互窗口，方便你手动旋转视角。
#
# 后面想换数据，只改 HARPNUM / T_REC。
# 后面想换图，只改 Visualization settings 区。

PYTHON_NOTE = r'Use G:\python_projects\envs\WPy64-31241\python-3.12.4.amd64\python.exe'

# ----- Data target -----
JSOC_EMAIL = "liusiwei21@mails.ucas.ac.cn"
HARPNUM = 377
T_REC = "2011.02.15_00:00:00_TAI"

# ----- Paths -----
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "sharp_cea"
OUTPUT_DIR = PROJECT_DIR / "output" / f"harp{HARPNUM}_{T_REC.replace(':', '').replace('.', '').replace('_TAI', '')}"

DOWNLOAD_IF_MISSING = True

# ----- PC-friendly grid settings -----
# A good first run for a normal desktop. Raise MAX_XY_POINTS to 160-220 and
# NLFFF_ITERATIONS to 1500-5000 after the demo behaves well.
MAX_XY_POINTS = 120
Z_SIZE = 72
NLFFF_ITERATIONS = 450
SAVE_EVERY = 25

# ----- Boundary conditioning -----
# The transverse field is noisy and photospheric data are
# not force-free; gentle smoothing and flux balancing make the demo much calmer.
CROP_TO_ACTIVE_REGION = True
CROP_THRESHOLD = 0.12
CROP_MARGIN_PIXELS = 70
GAUSSIAN_SIGMA_PIXELS = 0.7
SUBTRACT_MEAN_BZ = True
TRANSVERSE_SCALE = 1.0

# ----- Optimization settings -----
INITIAL_DT = 2.5e-4
MIN_DT_FACTOR = 1.0e-5
DT_GROWTH = 1.015
DT_SHRINK = 0.5
WEIGHT_TAPER_CELLS = 10
B_EPS = 1.0e-8

# ----- Visualization settings -----
# Set SHOW_INTERACTIVE_WINDOWS=True for double-click runs. Screenshots are saved
# either way; interactive windows let you rotate, zoom, and inspect the scene.
PYVISTA_OFF_SCREEN = False
SHOW_INTERACTIVE_WINDOWS = True
INTERACTIVE_VIEW_TARGETS = ("nlfff",)  # choose from: "potential", "nlfff"

BACKGROUND_COLOR = "#eef2f7"
TEXT_COLOR = "#101418"
MAGNETOGRAM_CMAP = "gray"
MAGNETOGRAM_CLIP_GAUSS = 1500.0

FIELDLINE_SEED_MODE = "pil_shear"
FIELDLINE_COLOR_MODE = "j_over_b"
FIELDLINE_CMAP = "turbo"
FIELDLINE_SIGNED_CMAP = "coolwarm"
FIELDLINE_SOLID_COLOR = "#0b1220"
FIELDLINE_SCALAR_PERCENTILES = (2.0, 98.0)
FIELDLINE_TUBE_RADIUS = 0.08
FIELDLINE_SEEDS = 220
FIELDLINE_MAX_LENGTH = 180.0
FIELDLINE_STEP = 0.35
FIELDLINE_SEED_Z = 1.2
FIELDLINE_RANDOM_SEED = 42

# Seed modes:
#   pil_shear         high horizontal field close to the polarity inversion line
#   strong_bz         strongest vertical-field footpoints
#   strong_bh         strongest transverse-field footpoints
#   mixed_bz_bh       strong vertical and transverse field together
#   polarity_balanced equal count from positive and negative polarities
#   grid              uniform seeds, useful for debugging connectivity
#   j_over_b_low      seeds from low-corona current proxy after extrapolation
#   random_weighted   reproducible weighted random seeds near the shear channel
#
# Color modes:
#   j_over_b          relative current strength, common for current channels
#   bmag              magnetic-field strength
#   height            geometric height
#   alpha             signed force-free alpha proxy, J.B/B^2
#   current_helicity  signed current-helicity proxy, J.B
#   solid             one clean color, no colorbar


@dataclass
class Boundary:
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    header: fits.Header
    cdelt_deg: float
    source_files: dict[str, Path]


@dataclass
class Diagnostics:
    iteration: int
    objective: float
    cw_sin: float
    div_l1: float
    dt: float


def jsoc_query() -> str:
    return f"hmi.sharp_cea_720s[{HARPNUM}][{T_REC}]{{Br,Bp,Bt}}"


def download_sharp_cea() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = find_component_files(DATA_DIR)
    if len(existing) == 3 or not DOWNLOAD_IF_MISSING:
        return

    print(f"Downloading JSOC SHARP CEA data: {jsoc_query()}")
    client = drms.Client(email=JSOC_EMAIL)
    request = client.export(jsoc_query(), method="url", protocol="fits")
    request.wait(timeout=180)
    if request.status != 0:
        raise RuntimeError(f"JSOC export failed with status {request.status}")
    request.download(str(DATA_DIR))


def find_component_files(directory: Path) -> dict[str, Path]:
    files = {}
    for component in ("Bp", "Bt", "Br"):
        matches = sorted(directory.glob(f"hmi.sharp_cea_720s.{HARPNUM}.*.{component}.fits"))
        if matches:
            files[component] = matches[-1]
    return files


def read_image_hdu(path: Path) -> tuple[np.ndarray, fits.Header]:
    hdul = fits.open(path)
    hdu = hdul[1] if len(hdul) > 1 else hdul[0]
    data = np.asarray(hdu.data, dtype=np.float64)
    header = hdu.header.copy()
    hdul.close()
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), header


def crop_bbox_from_bz(bz_img: np.ndarray) -> tuple[slice, slice]:
    if not CROP_TO_ACTIVE_REGION:
        return slice(0, bz_img.shape[0]), slice(0, bz_img.shape[1])
    peak = float(np.nanmax(np.abs(bz_img)))
    if peak <= 0:
        return slice(0, bz_img.shape[0]), slice(0, bz_img.shape[1])

    mask = np.abs(bz_img) >= peak * CROP_THRESHOLD
    yy, xx = np.where(mask)
    if yy.size == 0:
        return slice(0, bz_img.shape[0]), slice(0, bz_img.shape[1])

    y0 = max(0, int(yy.min()) - CROP_MARGIN_PIXELS)
    y1 = min(bz_img.shape[0], int(yy.max()) + CROP_MARGIN_PIXELS + 1)
    x0 = max(0, int(xx.min()) - CROP_MARGIN_PIXELS)
    x1 = min(bz_img.shape[1], int(xx.max()) + CROP_MARGIN_PIXELS + 1)
    return slice(y0, y1), slice(x0, x1)


def resize_image(img: np.ndarray, max_points: int) -> np.ndarray:
    ny, nx = img.shape
    scale = max(nx, ny) / float(max_points)
    if scale <= 1.0:
        return img.copy()
    new_nx = max(8, int(round(nx / scale)))
    new_ny = max(8, int(round(ny / scale)))
    return cv2.resize(img, (new_nx, new_ny), interpolation=cv2.INTER_AREA)


def load_boundary() -> Boundary:
    download_sharp_cea()
    files = find_component_files(DATA_DIR)
    missing = {"Bp", "Bt", "Br"} - set(files)
    if missing:
        raise FileNotFoundError(f"Missing SHARP CEA components in {DATA_DIR}: {sorted(missing)}")

    bp_img, header = read_image_hdu(files["Bp"])
    bt_img, _ = read_image_hdu(files["Bt"])
    br_img, _ = read_image_hdu(files["Br"])

    y_slice, x_slice = crop_bbox_from_bz(br_img)
    bx_img = bp_img[y_slice, x_slice]
    by_img = -bt_img[y_slice, x_slice]
    bz_img = br_img[y_slice, x_slice]

    bx_img = resize_image(bx_img, MAX_XY_POINTS)
    by_img = resize_image(by_img, MAX_XY_POINTS)
    bz_img = resize_image(bz_img, MAX_XY_POINTS)

    if GAUSSIAN_SIGMA_PIXELS > 0:
        bx_img = gaussian_filter(bx_img, GAUSSIAN_SIGMA_PIXELS)
        by_img = gaussian_filter(by_img, GAUSSIAN_SIGMA_PIXELS)
        bz_img = gaussian_filter(bz_img, GAUSSIAN_SIGMA_PIXELS * 0.5)

    if SUBTRACT_MEAN_BZ:
        bz_img = bz_img - np.mean(bz_img)

    bx_img = bx_img * TRANSVERSE_SCALE
    by_img = by_img * TRANSVERSE_SCALE

    # Internal convention is (x, y), while FITS images are (row/y, col/x).
    bx = np.ascontiguousarray(bx_img.T)
    by = np.ascontiguousarray(by_img.T)
    bz = np.ascontiguousarray(bz_img.T)
    cdelt = float(header.get("CDELT1", 0.03))
    return Boundary(bx=bx, by=by, bz=bz, header=header, cdelt_deg=cdelt, source_files=files)


def normalize_boundary(boundary: Boundary) -> tuple[Boundary, float]:
    bmax = float(np.nanmax(np.abs(boundary.bz)))
    if bmax <= 0:
        raise ValueError("Boundary Br is zero everywhere.")
    scaled = Boundary(
        bx=boundary.bx / bmax,
        by=boundary.by / bmax,
        bz=boundary.bz / bmax,
        header=boundary.header,
        cdelt_deg=boundary.cdelt_deg,
        source_files=boundary.source_files,
    )
    return scaled, bmax


def potential_field_fft(bz0: np.ndarray, nz: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = bz0.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(kxg * kxg + kyg * kyg)
    safe_k = np.where(k == 0.0, 1.0, k)

    bz_hat0 = np.fft.fft2(bz0)
    bz_hat0[0, 0] = 0.0

    bx = np.zeros((nx, ny, nz), dtype=np.float64)
    by = np.zeros_like(bx)
    bz = np.zeros_like(bx)

    for iz in range(nz):
        decay = np.exp(-k * float(iz))
        bz_hat = bz_hat0 * decay
        bx_hat = -1j * kxg / safe_k * bz_hat
        by_hat = -1j * kyg / safe_k * bz_hat
        bx_hat[k == 0.0] = 0.0
        by_hat[k == 0.0] = 0.0
        bx[:, :, iz] = np.real(np.fft.ifft2(bx_hat))
        by[:, :, iz] = np.real(np.fft.ifft2(by_hat))
        bz[:, :, iz] = np.real(np.fft.ifft2(bz_hat))

    return bx, by, bz


def grad(a: np.ndarray, axis: int) -> np.ndarray:
    return np.gradient(a, 1.0, axis=axis, edge_order=2)


def curl(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        grad(bz, 1) - grad(by, 2),
        grad(bx, 2) - grad(bz, 0),
        grad(by, 0) - grad(bx, 1),
    )


def divergence(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    return grad(bx, 0) + grad(by, 1) + grad(bz, 2)


def cross(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def make_weight(shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    nb = int(min(WEIGHT_TAPER_CELLS, max(1, nx // 4), max(1, ny // 4), max(1, nz // 4)))
    w = np.ones(shape, dtype=np.float64)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                distance = min(ix, nx - 1 - ix, iy, ny - 1 - iy, nz - 1 - iz)
                if distance < nb:
                    w[ix, iy, iz] = 0.5 * (1.0 - math.cos(math.pi * distance / nb))
    w[:, :, 0] = np.maximum(w[:, :, 0], 0.25)
    return w


def objective_and_force(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray], Diagnostics]:
    jx, jy, jz = curl(bx, by, bz)
    div_b = divergence(bx, by, bz)
    jxbx, jxby, jxbz = cross(jx, jy, jz, bx, by, bz)

    b2 = bx * bx + by * by + bz * bz + B_EPS
    omega_x = (jxbx - div_b * bx) / b2
    omega_y = (jxby - div_b * by) / b2
    omega_z = (jxbz - div_b * bz) / b2
    omega_x *= weight
    omega_y *= weight
    omega_z *= weight
    omega2 = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z

    tx, ty, tz = cross(omega_x, omega_y, omega_z, bx, by, bz)
    fx, fy, fz = curl(tx, ty, tz)

    tx, ty, tz = cross(omega_x, omega_y, omega_z, jx, jy, jz)
    fx -= tx
    fy -= ty
    fz -= tz

    dot_ob = omega_x * bx + omega_y * by + omega_z * bz
    fx -= grad(dot_ob, 0)
    fy -= grad(dot_ob, 1)
    fz -= grad(dot_ob, 2)

    fx += omega_x * div_b + omega2 * bx
    fy += omega_y * div_b + omega2 * by
    fz += omega_z * div_b + omega2 * bz

    fx *= weight
    fy *= weight
    fz *= weight

    objective = float(np.mean(weight * ((jxbx * jxbx + jxby * jxby + jxbz * jxbz) / b2 + div_b * div_b)))
    jmag = np.sqrt(jx * jx + jy * jy + jz * jz)
    bmag = np.sqrt(b2)
    cw_sin = float(np.sum(np.sqrt(jxbx * jxbx + jxby * jxby + jxbz * jxbz)) / (np.sum(jmag * bmag) + B_EPS))
    div_l1 = float(np.mean(np.abs(div_b)) / (np.mean(bmag) + B_EPS))
    diag = Diagnostics(iteration=0, objective=objective, cw_sin=cw_sin, div_l1=div_l1, dt=0.0)
    return objective, (fx, fy, fz), diag


def run_nlfff(boundary: Boundary, potential: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Diagnostics]]:
    bx, by, bz = [a.copy() for a in potential]
    nx, ny, nz = bx.shape
    bx[:, :, 0] = boundary.bx
    by[:, :, 0] = boundary.by
    bz[:, :, 0] = boundary.bz

    weight = make_weight(bx.shape)
    diagnostics: list[Diagnostics] = []
    dt = INITIAL_DT
    objective, force, diag = objective_and_force(bx, by, bz, weight)
    diagnostics.append(Diagnostics(0, objective, diag.cw_sin, diag.div_l1, dt))

    inner = np.s_[1 : nx - 1, 1 : ny - 1, 1 : nz - 1]
    for iteration in trange(1, NLFFF_ITERATIONS + 1, desc="NLFFF optimization"):
        fx, fy, fz = force
        old = (bx.copy(), by.copy(), bz.copy())
        local_dt = dt
        accepted = False

        while local_dt >= INITIAL_DT * MIN_DT_FACTOR:
            bx[inner] = old[0][inner] + local_dt * fx[inner]
            by[inner] = old[1][inner] + local_dt * fy[inner]
            bz[inner] = old[2][inner] + local_dt * fz[inner]
            bx[:, :, 0] = boundary.bx
            by[:, :, 0] = boundary.by
            bz[:, :, 0] = boundary.bz

            trial_objective, trial_force, trial_diag = objective_and_force(bx, by, bz, weight)
            if np.isfinite(trial_objective) and trial_objective <= objective:
                objective = trial_objective
                force = trial_force
                dt = local_dt * DT_GROWTH
                accepted = True
                diag = trial_diag
                break

            bx[:], by[:], bz[:] = old
            local_dt *= DT_SHRINK

        if not accepted:
            print(f"Stopped: no decreasing step at iteration {iteration}.")
            break

        if iteration % SAVE_EVERY == 0 or iteration == NLFFF_ITERATIONS:
            diagnostics.append(Diagnostics(iteration, objective, diag.cw_sin, diag.div_l1, dt))

    return bx, by, bz, diagnostics


def stack_for_fits(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    return np.stack([np.moveaxis(bx, 2, 0), np.moveaxis(by, 2, 0), np.moveaxis(bz, 2, 0)])


def write_field_fits(path: Path, bx: np.ndarray, by: np.ndarray, bz: np.ndarray, boundary: Boundary, bscale: float, field_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(stack_for_fits(bx * bscale, by * bscale, bz * bscale))
    hdr = hdu.header
    hdr["FIELD"] = field_type
    hdr["AXES"] = "component,z,y,x"
    hdr["COMP0"] = "Bx=Bp"
    hdr["COMP1"] = "By=-Bt"
    hdr["COMP2"] = "Bz=Br"
    hdr["BUNIT"] = "Gauss"
    hdr["HARPNUM"] = HARPNUM
    hdr["T_REC"] = T_REC
    hdr["CDELT1"] = boundary.cdelt_deg
    hdr["SOURCE"] = "HMI SHARP CEA 720s"
    hdu.writeto(path, overwrite=True)


def write_diagnostics(path: Path, diagnostics: list[Diagnostics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective", "cw_sin", "div_l1", "dt"])
        for d in diagnostics:
            writer.writerow([d.iteration, f"{d.objective:.10e}", f"{d.cw_sin:.10e}", f"{d.div_l1:.10e}", f"{d.dt:.10e}"])


def j_over_b(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    jx, jy, jz = curl(bx, by, bz)
    jmag = np.sqrt(jx * jx + jy * jy + jz * jz)
    bmag = np.sqrt(bx * bx + by * by + bz * bz)
    return jmag / (bmag + B_EPS)


def fieldline_scalar(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> tuple[str | None, np.ndarray | None, str, str, bool]:
    mode = FIELDLINE_COLOR_MODE.lower()
    jx, jy, jz = curl(bx, by, bz)
    b2 = bx * bx + by * by + bz * bz + B_EPS
    bmag = np.sqrt(b2)

    if mode == "solid":
        return None, None, "", FIELDLINE_CMAP, False
    if mode == "j_over_b":
        jmag = np.sqrt(jx * jx + jy * jy + jz * jz)
        return "J_over_B", jmag / bmag, "|J|/|B|", FIELDLINE_CMAP, False
    if mode == "bmag":
        return "Bmag", bmag, "|B|", FIELDLINE_CMAP, False
    if mode == "height":
        height = np.broadcast_to(np.arange(bx.shape[2], dtype=np.float64)[None, None, :], bx.shape)
        return "height", height, "height", FIELDLINE_CMAP, False
    if mode == "alpha":
        alpha = (jx * bx + jy * by + jz * bz) / b2
        return "alpha", alpha, "alpha = J.B/B^2", FIELDLINE_SIGNED_CMAP, True
    if mode == "current_helicity":
        helicity = jx * bx + jy * by + jz * bz
        return "current_helicity", helicity, "J.B", FIELDLINE_SIGNED_CMAP, True
    raise ValueError(
        "FIELDLINE_COLOR_MODE must be one of: solid, j_over_b, bmag, height, alpha, current_helicity"
    )


def robust_clim(values: np.ndarray, signed: bool) -> tuple[float, float]:
    data = np.asarray(values)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(data, FIELDLINE_SCALAR_PERCENTILES)
    if signed:
        lim = max(abs(float(lo)), abs(float(hi)), B_EPS)
        return -lim, lim
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def build_grid(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> pv.ImageData:
    nx, ny, nz = bx.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    vectors = np.column_stack([bx.ravel(order="F"), by.ravel(order="F"), bz.ravel(order="F")])
    grid.point_data["B"] = vectors
    scalar_name, scalar_values, _, _, _ = fieldline_scalar(bx, by, bz)
    if scalar_name is not None and scalar_values is not None:
        grid.point_data[scalar_name] = scalar_values.ravel(order="F")
    grid.set_active_vectors("B")
    return grid


def top_score_points(score: np.ndarray, n: int) -> np.ndarray:
    flat = np.nan_to_num(score, nan=-np.inf).ravel()
    n = min(n, flat.size)
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    ids = np.argpartition(flat, -n)[-n:]
    xs, ys = np.unravel_index(ids, score.shape)
    return np.column_stack([xs.astype(float), ys.astype(float), np.ones(n) * FIELDLINE_SEED_Z])


def seed_points(boundary: Boundary, n: int, field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    bx, by, bz = boundary.bx, boundary.by, boundary.bz
    bh = np.sqrt(bx * bx + by * by)
    mode = FIELDLINE_SEED_MODE.lower()

    if mode == "pil_shear":
        score = bh * np.exp(-np.abs(bz) / (0.18 * np.max(np.abs(bz)) + B_EPS))
        return top_score_points(score, n)
    if mode == "strong_bz":
        return top_score_points(np.abs(bz), n)
    if mode == "strong_bh":
        return top_score_points(bh, n)
    if mode == "mixed_bz_bh":
        return top_score_points(np.abs(bz) * bh, n)
    if mode == "polarity_balanced":
        pos = top_score_points(np.where(bz > 0, np.abs(bz), -np.inf), n // 2)
        neg = top_score_points(np.where(bz < 0, np.abs(bz), -np.inf), n - len(pos))
        return np.vstack([pos, neg])
    if mode == "grid":
        nx, ny = bz.shape
        step = max(2, int(round(math.sqrt((nx * ny) / max(n, 1)))))
        xs, ys = np.meshgrid(np.arange(1, nx - 1, step), np.arange(1, ny - 1, step), indexing="ij")
        points = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
        return np.column_stack([points[:n], np.ones(min(n, len(points))) * FIELDLINE_SEED_Z])
    if mode == "j_over_b_low":
        if field is None:
            return top_score_points(bh, n)
        scalar = j_over_b(*field)
        iz = min(2, scalar.shape[2] - 1)
        return top_score_points(scalar[:, :, iz], n)
    if mode == "random_weighted":
        score = np.maximum(bh * np.exp(-np.abs(bz) / (0.22 * np.max(np.abs(bz)) + B_EPS)), 0.0)
        weights = score.ravel()
        weights = weights / (np.sum(weights) + B_EPS)
        rng = np.random.default_rng(FIELDLINE_RANDOM_SEED)
        ids = rng.choice(weights.size, size=min(n, weights.size), replace=False, p=weights)
        xs, ys = np.unravel_index(ids, score.shape)
        return np.column_stack([xs.astype(float), ys.astype(float), np.ones(len(ids)) * FIELDLINE_SEED_Z])
    raise ValueError(
        "FIELDLINE_SEED_MODE must be one of: pil_shear, strong_bz, strong_bh, mixed_bz_bh, "
        "polarity_balanced, grid, j_over_b_low, random_weighted"
    )


def make_pyvista_view(
    path_prefix: Path,
    boundary: Boundary,
    field: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    view_target: str,
) -> None:
    bx, by, bz = field
    grid = build_grid(bx, by, bz)
    scalar_name, scalar_values, scalar_title, scalar_cmap, signed_scalar = fieldline_scalar(bx, by, bz)
    seeds = pv.PolyData(seed_points(boundary, FIELDLINE_SEEDS, field))
    lines = grid.streamlines_from_source(
        seeds,
        vectors="B",
        integration_direction="both",
        max_length=FIELDLINE_MAX_LENGTH,
        initial_step_length=FIELDLINE_STEP,
        terminal_speed=1.0e-6,
    )

    nx, ny, _ = bx.shape
    xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
    zz = np.zeros_like(xx, dtype=float)
    plane = pv.StructuredGrid(xx, yy, zz)
    br_norm = boundary.bz / (np.nanmax(np.abs(boundary.bz)) + B_EPS)
    plane.point_data["Br"] = np.clip(br_norm, -1.0, 1.0).ravel(order="F")

    pv.OFF_SCREEN = PYVISTA_OFF_SCREEN
    plotter = pv.Plotter(off_screen=PYVISTA_OFF_SCREEN, window_size=(1500, 1100))
    plotter.set_background(BACKGROUND_COLOR)
    plotter.add_mesh(plane, scalars="Br", cmap=MAGNETOGRAM_CMAP, clim=(-1, 1), show_scalar_bar=False)
    if lines.n_points > 0:
        tubes = lines.tube(radius=FIELDLINE_TUBE_RADIUS)
        if scalar_name is None:
            plotter.add_mesh(tubes, color=FIELDLINE_SOLID_COLOR, show_scalar_bar=False)
        else:
            clim = robust_clim(scalar_values, signed_scalar)
            plotter.add_mesh(
                tubes,
                scalars=scalar_name,
                cmap=scalar_cmap,
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": scalar_title,
                    "vertical": True,
                    "position_x": 0.90,
                    "position_y": 0.18,
                    "width": 0.06,
                    "height": 0.64,
                    "title_font_size": 16,
                    "label_font_size": 12,
                    "color": TEXT_COLOR,
                },
            )
    plotter.add_text(title, position="upper_left", font_size=13, color=TEXT_COLOR)
    plotter.camera_position = [(nx * 0.55, -ny * 1.45, max(nx, ny) * 0.9), (nx * 0.52, ny * 0.48, 18), (0, 0, 1)]
    plotter.enable_anti_aliasing()

    screenshot = path_prefix.with_suffix(".png")
    plotter.screenshot(str(screenshot))
    try:
        plotter.export_html(str(path_prefix.with_suffix(".html")))
    except Exception as exc:
        print(f"HTML export skipped: {exc}")
    if SHOW_INTERACTIVE_WINDOWS and view_target in INTERACTIVE_VIEW_TARGETS and not PYVISTA_OFF_SCREEN:
        plotter.show(title=title)
    plotter.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    boundary_raw = load_boundary()
    boundary, bscale = normalize_boundary(boundary_raw)
    nz = min(Z_SIZE, max(8, min(boundary.bx.shape)))

    potential = potential_field_fft(boundary.bz, nz=nz)
    nlfff = run_nlfff(boundary, potential)
    nlfff_field = nlfff[:3]
    diagnostics = nlfff[3]

    write_field_fits(OUTPUT_DIR / "potential_field.fits", *potential, boundary_raw, bscale, "potential")
    write_field_fits(OUTPUT_DIR / "nlfff_field.fits", *nlfff_field, boundary_raw, bscale, "optimization_nlfff")
    write_diagnostics(OUTPUT_DIR / "diagnostics.csv", diagnostics)
    make_pyvista_view(
        OUTPUT_DIR / "potential_fieldlines",
        boundary,
        potential,
        "Potential field: NOAA AR 11158 / HARP 377",
        "potential",
    )
    make_pyvista_view(
        OUTPUT_DIR / "nlfff_fieldlines",
        boundary,
        nlfff_field,
        "Optimization NLFFF: NOAA AR 11158 / HARP 377",
        "nlfff",
    )
    print("Done.")


if __name__ == "__main__":
    main()
