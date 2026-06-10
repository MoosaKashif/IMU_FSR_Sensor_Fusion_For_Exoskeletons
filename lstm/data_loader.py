"""
lstm/data_loader.py

Loads all .mat files from the four dataset sub-folders, aligns them on a
shared time axis by interpolation, concatenates into a single feature
vector per timestep, and returns fixed-length windowed sequences.

Feature layout (18 features):
  [0]   p1       — IMU_data: pitch angle IMU1
  [1]   p2       — IMU_data: pitch angle IMU2
  [2]   pavg     — IMU_data: average pitch
  [3]   ax1      — IMU_accel: X acceleration IMU1
  [4]   ay1      — IMU_accel: Y acceleration IMU1
  [5]   az1      — IMU_accel: Z acceleration IMU1
  [6]   ax2      — IMU_accel: X acceleration IMU2
  [7]   ay2      — IMU_accel: Y acceleration IMU2
  [8]   az2      — IMU_accel: Z acceleration IMU2
  [9]   dx1      — IMU_derivatives: dX IMU1
  [10]  dy1      — IMU_derivatives: dY IMU1
  [11]  dz1      — IMU_derivatives: dZ IMU1
  [12]  dx2      — IMU_derivatives: dX IMU2
  [13]  dy2      — IMU_derivatives: dY IMU2
  [14]  dz2      — IMU_derivatives: dZ IMU2
  [15]  fsr1     — FSR_data: FSR sensor 1  (zeros if absent)
  [16]  fsr2     — FSR_data: FSR sensor 2  (zeros if absent)
  [17]  fsr3     — FSR_data: FSR sensor 3  (zeros if absent)

Labels are assigned via uniform quantile segmentation of each recording's
average pitch envelope into 4 gait phases:
  0 — Heel Strike
  1 — Stance
  2 — Push-Off
  3 — Swing
"""

import os
import glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
# PATHS  (resolved relative to this file's
#         parent = project root)
# ─────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATHS = {
    "imu":    os.path.join(_ROOT, "dataset", "IMU_data"),
    "accel":  os.path.join(_ROOT, "dataset", "IMU_accel"),
    "deriv":  os.path.join(_ROOT, "dataset", "IMU_derivatives"),
    "fsr":    os.path.join(_ROOT, "dataset", "FSR_data"),
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_FEATURES   = 18       # total feature width
N_CLASSES    = 4        # gait phases
WINDOW_SIZE  = 100      # timesteps per sequence
WINDOW_STEP  = 50       # stride  (50 % overlap)
N_SAMPLES    = 20       # recordings per modality


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _safe_load(path: str) -> dict | None:
    """Load a .mat file, returning None on failure."""
    try:
        return sio.loadmat(path)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def _interp_to(t_src: np.ndarray,
               sig: np.ndarray,
               t_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate sig (defined on t_src) onto t_dst."""
    f = interp1d(t_src, sig,
                 kind="linear",
                 bounds_error=False,
                 fill_value=(sig[0], sig[-1]))
    return f(t_dst)


def _assign_labels(pavg: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """
    Divide each recording into n_classes gait phases by
    quantile-based segmentation of the average-pitch signal.

    Produces per-timestep integer labels 0..n_classes-1.
    """
    labels = np.zeros(len(pavg), dtype=np.int64)
    thresholds = np.percentile(pavg, np.linspace(0, 100, n_classes + 1))
    for i in range(n_classes):
        lo = thresholds[i]
        hi = thresholds[i + 1]
        mask = (pavg >= lo) & (pavg <= hi)
        labels[mask] = i
    return labels


def _load_sample(idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load a single recording (by 1-based index), returning
    (features, labels) where:
      features : (T, N_FEATURES) float32
      labels   : (T,)            int64

    Returns None if critical IMU data is missing.
    """

    # ── IMU angles (required) ─────────────────
    imu_path = os.path.join(DATASET_PATHS["imu"],
                            f"IMU_data_{idx}.mat")
    imu = _safe_load(imu_path)
    if imu is None or "t" not in imu:
        print(f"  [SKIP] Sample {idx}: missing IMU_data")
        return None

    t_ref  = imu["t"].flatten().astype(float)
    p1     = imu["p1"].flatten().astype(float)
    p2     = imu["p2"].flatten().astype(float)
    pavg   = imu["pavg"].flatten().astype(float)

    T = len(t_ref)

    # ── IMU accel (required) ──────────────────
    accel_path = os.path.join(DATASET_PATHS["accel"],
                              f"IMU_accel_{idx}.mat")
    accel = _safe_load(accel_path)
    if accel is not None and "ax1" in accel:
        t_a  = accel["t"].flatten().astype(float)
        ax1  = _interp_to(t_a, accel["ax1"].flatten(), t_ref)
        ay1  = _interp_to(t_a, accel["ay1"].flatten(), t_ref)
        az1  = _interp_to(t_a, accel["az1"].flatten(), t_ref)
        ax2  = _interp_to(t_a, accel["ax2"].flatten(), t_ref)
        ay2  = _interp_to(t_a, accel["ay2"].flatten(), t_ref)
        az2  = _interp_to(t_a, accel["az2"].flatten(), t_ref)
    else:
        print(f"  [WARN] Sample {idx}: IMU_accel missing — using zeros")
        ax1 = ay1 = az1 = ax2 = ay2 = az2 = np.zeros(T)

    # ── IMU derivatives (optional) ────────────
    deriv_path = os.path.join(DATASET_PATHS["deriv"],
                              f"IMU_derivatives_{idx}.mat")
    deriv = _safe_load(deriv_path)
    if deriv is not None and "dx1" in deriv:
        t_d  = deriv["t"].flatten().astype(float)
        dx1  = _interp_to(t_d, deriv["dx1"].flatten(), t_ref)
        dy1  = _interp_to(t_d, deriv["dy1"].flatten(), t_ref)
        dz1  = _interp_to(t_d, deriv["dz1"].flatten(), t_ref)
        dx2  = _interp_to(t_d, deriv["dx2"].flatten(), t_ref)
        dy2  = _interp_to(t_d, deriv["dy2"].flatten(), t_ref)
        dz2  = _interp_to(t_d, deriv["dz2"].flatten(), t_ref)
    else:
        print(f"  [WARN] Sample {idx}: IMU_derivatives missing — using zeros")
        dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = np.zeros(T)

    # ── FSR (optional — zeros if absent) ─────
    fsr_path = os.path.join(DATASET_PATHS["fsr"],
                            f"FSR_data_{idx}.mat")
    fsr = _safe_load(fsr_path)
    if fsr is not None and "fsr1" in fsr:
        t_f  = fsr["t"].flatten().astype(float)
        fsr1 = _interp_to(t_f, fsr["fsr1"].flatten(), t_ref)
        fsr2 = _interp_to(t_f, fsr["fsr2"].flatten(), t_ref)
        fsr3 = _interp_to(t_f, fsr["fsr3"].flatten(), t_ref)
    else:
        print(f"  [INFO] Sample {idx}: FSR absent — padding with zeros")
        fsr1 = fsr2 = fsr3 = np.zeros(T)

    # ── Concatenate feature matrix ────────────
    features = np.stack([
        p1, p2, pavg,
        ax1, ay1, az1, ax2, ay2, az2,
        dx1, dy1, dz1, dx2, dy2, dz2,
        fsr1, fsr2, fsr3
    ], axis=1).astype(np.float32)   # (T, 18)

    # ── Labels ───────────────────────────────
    labels = _assign_labels(pavg)

    return features, labels


def _sliding_windows(features: np.ndarray,
                     labels: np.ndarray,
                     window: int = WINDOW_SIZE,
                     step: int   = WINDOW_STEP
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a (T, F) recording into overlapping windows.
    Label per window = majority class among the window timesteps.
    """
    X_list, y_list = [], []
    T = len(features)
    start = 0
    while start + window <= T:
        window_feats  = features[start: start + window]   # (W, F)
        window_labels = labels[start: start + window]     # (W,)
        label = int(np.bincount(window_labels).argmax())
        X_list.append(window_feats)
        y_list.append(label)
        start += step
    if not X_list:
        return np.empty((0, window, features.shape[1])), np.empty((0,), dtype=int)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def load_dataset(verbose: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all 20 recordings, window them, and return (X, y).

      X : (N_windows, WINDOW_SIZE, N_FEATURES)  float32
      y : (N_windows,)                           int64  0..3
    """
    X_all, y_all = [], []

    for idx in range(1, N_SAMPLES + 1):
        if verbose:
            print(f"\n[LOAD] Recording {idx}/{N_SAMPLES}")
        result = _load_sample(idx)
        if result is None:
            continue
        features, labels = result
        X_win, y_win = _sliding_windows(features, labels)
        if len(X_win) == 0:
            print(f"  [WARN] Sample {idx}: no windows produced (too short?)")
            continue
        X_all.append(X_win)
        y_all.append(y_win)
        if verbose:
            print(f"  shape={features.shape}  windows={len(X_win)}")

    if not X_all:
        raise RuntimeError("No data was loaded — check your dataset paths.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y
