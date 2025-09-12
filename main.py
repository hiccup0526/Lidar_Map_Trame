from __future__ import annotations
"""
KITTI LiDAR viewer + simple SLAM (ICP + scan-context) — RT-optimized
Key performance changes:
- SLAM thread no longer builds map arrays; it only marks the map "dirty"
- VoxelMap supports incremental updates (dirty keys -> array rows) to avoid full rebuilds
- Map vtkPolyData is persistent; per-publish we update arrays in-place
- Publish is throttled by map_publish_every_n and adaptive stride
- Heavier default voxels / lazier global opt / coarser loop candidate search
"""

import argparse, platform, threading, time, multiprocessing, os, math
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, multiprocessing.cpu_count())))

import numpy as np
import vtk

# ---------- Cupoch (GPU ICP) availability & runtime toggle ----------
HAVE_CUPOCH = False
try:
    import cupoch as cph
    HAVE_CUPOCH = True
except Exception:
    cph = None  # type: ignore

import open3d as o3d  # CPU fallback

from concurrent.futures import ThreadPoolExecutor
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtkw
from trame.widgets import html as h

import traceback

IS_WINDOWS = platform.system() == "Windows"

# --------- VTK <-> NumPy helpers ---------
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
except Exception:
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy  # type: ignore

# --------- small coercion utils ---------
def as_int(v, default=0):
    try: return int(float(v))
    except Exception: return default

def as_float(v, default=0.0):
    try: return float(v)
    except Exception: return default

LOG_LEVEL = 2   # 1 = verbose, 2 = silent
client_connected = {"v": False}

def log(msg, level=1, **kwargs):
    """
    Log messages depending on LOG_LEVEL.
    level=1 → printed if LOG_LEVEL == 1
    level=2 → printed if LOG_LEVEL == 1 (but suppressed if LOG_LEVEL > 1)
    """
    global LOG_LEVEL
    if LOG_LEVEL <= level:
        print(msg, **kwargs)

# --------- IO & preprocessing ---------
def list_kitti_bins(seq_dir: str):
    p = Path(seq_dir)
    if not p.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    files = sorted(p.glob("*.bin")) or sorted(p.rglob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin files found under: {seq_dir}")
    return files

def load_kitti_bin(file_path: str) -> np.ndarray:
    arr = np.fromfile(file_path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Unexpected .bin size: {file_path}")
    return arr.reshape((-1, 4))

def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel is None or voxel <= 0:
        return points
    xyz = points[:, :3].astype(np.float32, copy=False)
    ijk = np.floor(xyz / float(voxel)).astype(np.int64)
    try:
        _, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    except TypeError:
        ijk_view = np.ascontiguousarray(ijk).view(np.dtype((np.void, ijk.dtype.itemsize * ijk.shape[1])))
        _, inv, counts = np.unique(ijk_view, return_inverse=True, return_counts=True)
    sums_xyz = np.zeros((counts.size, 3), dtype=np.float64)
    sums_i = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums_xyz, inv, xyz)
    np.add.at(sums_i, inv, points[:, 3])
    centroids = sums_xyz / counts[:, None]
    mean_i = (sums_i / counts).reshape(-1, 1)
    return np.hstack([centroids.astype(np.float32), mean_i.astype(np.float32)])

def crop_distance(points: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    if (min_d is None or min_d <= 0) and (max_d is None or max_d <= 0):
        return points
    d = np.linalg.norm(points[:, :3], axis=1)
    m = np.ones(len(points), dtype=bool)
    if min_d is not None and min_d > 0: m &= d >= min_d
    if max_d is not None and max_d > 0: m &= d <= max_d
    return points[m]

# --------- VTK conversion ---------
def np_points_to_polydata(points: np.ndarray, color_mode: str, i_win: Tuple[float, float], constant_rgb=None) -> vtk.vtkPolyData:
    n = len(points)
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    if n == 0:
        poly.SetPoints(pts)
        return poly

    pts.SetData(numpy_to_vtk(points[:, :3].copy(), deep=True))
    poly.SetPoints(pts)

    ids = np.arange(n, dtype=np.int64)
    cells = np.empty((n, 2), dtype=np.int64); cells[:, 0] = 1; cells[:, 1] = ids
    ca = numpy_to_vtkIdTypeArray(cells.ravel(order="C"), deep=True)
    verts = vtk.vtkCellArray(); verts.SetCells(n, ca)
    poly.SetVerts(verts)

    if constant_rgb is not None:
        colors = np.tile(np.asarray(constant_rgb, dtype=np.uint8), (n, 1))
    else:
        intens = points[:, 3] if points.shape[1] > 3 else np.zeros(n, dtype=np.float32)
        lo, hi = i_win
        if hi <= lo:
            lo = float(np.percentile(intens, 2.0)) if intens.size else 0.0
            hi = float(np.percentile(intens, 98.0)) if intens.size else 1.0
            if hi <= lo: hi = lo + 1.0
        t = np.clip((intens - lo) / (hi - lo), 0, 1)
        colors = turbo_colormap(t) if color_mode == "Turbo" else np.stack([(t*255).astype(np.uint8)]*3, axis=1)
    vtk_colors = numpy_to_vtk(colors.astype(np.uint8), deep=True)
    vtk_colors.SetName("RGB")
    poly.GetPointData().SetScalars(vtk_colors)
    return poly

def turbo_colormap(t: np.ndarray) -> np.ndarray:
    c = np.array([
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839,   4.84296658, -14.18503333,   4.27729857,  2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ])
    t = np.clip(np.asarray(t), 0.0, 1.0)
    r = np.polyval(c[0][::-1], t); g = np.polyval(c[1][::-1], t); b = np.polyval(c[2][::-1], t)
    rgb = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

def np_to_vtk_matrix(T: np.ndarray) -> vtk.vtkMatrix4x4:
    m = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(T[r, c]))
    return m

# --------- ICP backend ---------
USE_GPU = False

def _o3d_pc_from_np(pts_np: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    if pts_np.size:
        pc.points = o3d.utility.Vector3dVector(pts_np[:, :3].astype(np.float64, copy=False))
    return pc

def _ensure_normals(pc: o3d.geometry.PointCloud, radius: float, max_nn: int = 60):
    if len(pc.points) == 0:
        return pc
    if not pc.has_normals():
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        # any fixed point is fine for consistent orientation
        pc.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    return pc

def _safe_fpfh(pc: o3d.geometry.PointCloud, voxel: float):
    """Compute FPFH if possible; otherwise return None instead of throwing."""
    if len(pc.points) < 15:
        return None
    # FPFH requires normals:
    try:
        # normal radius ~ 2.5 * voxel; feature radius ~ 5 * voxel (clamped)
        nr = max(0.5, 2.5 * float(voxel)) if voxel and voxel > 0 else 1.0
        fr = max(1.0, 5.0 * float(voxel)) if voxel and voxel > 0 else 2.0
        _ensure_normals(pc, radius=nr, max_nn=64)
        return o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=100)
        )
    except Exception as e:
        print(f"[ICP ] skip FPFH: {e}")
        return None
    
def _o3d_make_loss(delta: float = 1.0):
    """Open3D ≥0.19: use Loss classes; otherwise return None (no robust loss)."""
    reg = o3d.pipelines.registration
    for name in ("TukeyLoss", "HuberLoss", "L2Loss"):
        if hasattr(reg, name):
            Loss = getattr(reg, name)
            try:
                return Loss(delta)   # Tukey/Huber take a scale
            except TypeError:
                try:
                    return Loss()    # L2Loss may be parameterless
                except Exception:
                    pass
    return None  # fine to run without a robust loss

def _o3d_trans_estimation(point_to_plane: bool, delta: float = 1.0):
    """Build the proper estimator for the current Open3D version."""
    reg = o3d.pipelines.registration
    loss = _o3d_make_loss(delta)
    if point_to_plane:
        try:
            return reg.TransformationEstimationPointToPlane(loss=loss) if loss else \
                   reg.TransformationEstimationPointToPlane()
        except TypeError:
            return reg.TransformationEstimationPointToPlane()
    else:
        try:
            return reg.TransformationEstimationPointToPoint(loss=loss) if loss else \
                   reg.TransformationEstimationPointToPoint()
        except TypeError:
            return reg.TransformationEstimationPointToPoint()

def _ensure_normals_019(pc: o3d.geometry.PointCloud, voxel: float = 0.5):
    # Minimal, 0.19-safe normal estimation
    if not pc.has_normals():
        r = max(1.5 * float(voxel), 0.6)
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30)
        )


def icp_transform_backend(src_pts_np: np.ndarray,
                          tgt_pts_np: np.ndarray,
                          voxel: float,
                          init=np.eye(4)):
    """
    Run ICP (GPU via Cupoch if available, else CPU via Open3D).
    - Prefers Point-to-Plane ICP (needs normals).
    - Falls back to Point-to-Point if normals or loss unavailable.
    """

    # ---------- GPU path (Cupoch) ----------
    if USE_GPU and HAVE_CUPOCH:
        try:
            src = cph.geometry.PointCloud()
            tgt = cph.geometry.PointCloud()
            src.points = cph.utility.Vector3fVector(src_pts_np[:, :3].astype(np.float32))
            tgt.points = cph.utility.Vector3fVector(tgt_pts_np[:, :3].astype(np.float32))

            if voxel and voxel > 0:
                src = src.voxel_down_sample(float(voxel))
                tgt = tgt.voxel_down_sample(float(voxel))

            max_corr = max(2.0 * float(voxel), 1.0) if voxel and voxel > 0 else 1.0
            result = cph.registration.registration_icp(
                src, tgt, max_corr, init.astype(np.float32),
                cph.registration.TransformationEstimationPointToPoint(),
                cph.registration.ICPConvergenceCriteria(max_iteration=25),
            )
            T = np.array(result.transformation, dtype=np.float32)
            fit = float(getattr(result, "fitness", 1.0))
            return T, fit
        except Exception as e:
            log(f"[ICP ] GPU ICP failed, falling back to CPU: {e}", level=1)

    # ---------- CPU path (Open3D) ----------
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts_np[:, :3].astype(np.float64))
    tgt.points = o3d.utility.Vector3dVector(tgt_pts_np[:, :3].astype(np.float64))

    if voxel and voxel > 0:
        src = src.voxel_down_sample(float(voxel))
        tgt = tgt.voxel_down_sample(float(voxel))

    max_corr = max(2.0 * float(voxel), 1.0) if voxel and voxel > 0 else 1.0

    # Try Point-to-Plane ICP if normals can be estimated
    try:
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr, init,
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25),
        )
        T = np.array(result.transformation, dtype=np.float32)
        fit = float(result.fitness)
        return T, fit
    except Exception as e:
        log(f"[ICP ] CPU P2L path failed, falling back to P2P: {e}", level=1)

    # Fallback: Point-to-Point ICP
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25),
    )
    return np.array(result.transformation, dtype=np.float32), float(result.fitness)

# --------- Scan-context ---------
def make_scan_context(points: np.ndarray, n_ring=20, n_sector=60, max_range=80.0) -> np.ndarray:
    xyz = points[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x*x + y*y)
    a = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    ring_idx = np.clip((r / max_range * n_ring).astype(np.int32), 0, n_ring - 1)
    sector_idx = np.clip((a * n_sector).astype(np.int32), 0, n_sector - 1)

    sc = np.zeros((n_ring, n_sector), dtype=np.float32)
    # flatten indices for fast max-by-bin
    flat_idx = ring_idx * n_sector + sector_idx
    # For duplicated bins, take max z
    # Sort by z so latest overwrites; or use bincount-based reduction:
    order = np.argsort(z)  # ascending, last wins
    flat_idx = flat_idx[order]; z_sorted = z[order]
    # write max via take/put – or use maximum.at on a 1D view
    sc_flat = sc.ravel()
    np.maximum.at(sc_flat, flat_idx, z_sorted)
    sc = sc_flat.reshape(n_ring, n_sector)
    # row-wise normalize
    row_max = sc.max(axis=1, keepdims=True) + 1e-6
    return sc / row_max

def scan_context_distance(sc1: np.ndarray, sc2: np.ndarray) -> float:
    n_ring, n_sector = sc1.shape
    best = 1e9
    for shift in range(0, n_sector, 1):
        sc2s = np.roll(sc2, shift, axis=1)
        num = (sc1 * sc2s).sum(axis=1)
        den = np.linalg.norm(sc1, axis=1) * np.linalg.norm(sc2s, axis=1) + 1e-9
        sim = (num / den).mean()
        dist = 1.0 - sim
        if dist < best: best = float(dist)
    return best

# --------- Incremental VoxelMap ---------
class VoxelMap:
    """Incrementally maintained voxel grid with dirty-key tracking."""
    def __init__(self, voxel=0.45):
        self.voxel = float(voxel)
        self.acc: Dict[int, List[float]] = {}   # key -> [sumx, sumy, sumz, sumi, count]
        self._dirty: set[int] = set()
        # arrays (lazy)
        self._arr: Optional[np.ndarray] = None
        self._keys: List[int] = []
        self._row: Dict[int, int] = {}

    def clear(self):
        self.acc.clear()
        self._dirty.clear()
        self._arr = None
        self._keys = []
        self._row = {}

    @staticmethod
    def _encode_keys(g: np.ndarray) -> np.ndarray:
        return (g[:, 0].astype(np.int64) << 42) ^ (g[:, 1].astype(np.int64) << 21) ^ g[:, 2].astype(np.int64)

    def insert(self, pts: np.ndarray, T: np.ndarray):
        if pts.size == 0:
            return
        Pw = (T @ np.c_[pts[:, :3], np.ones(len(pts))].T).T[:, :3]
        I = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts), np.float32)
        g = np.floor(Pw / self.voxel).astype(np.int64)
        keys = self._encode_keys(g)
        uk, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)

        sums_xyz = np.zeros((uk.size, 3), dtype=np.float64)
        sums_i   = np.zeros(uk.size, dtype=np.float64)
        np.add.at(sums_xyz, inv, Pw)
        np.add.at(sums_i,   inv, I)

        for k, c, sxyz, si in zip(uk.tolist(), counts.tolist(), sums_xyz, sums_i):
            if k in self.acc:
                a = self.acc[k]
                a[0] += sxyz[0]; a[1] += sxyz[1]; a[2] += sxyz[2]
                a[3] += float(si); a[4] += float(c)
            else:
                self.acc[k] = [float(sxyz[0]), float(sxyz[1]), float(sxyz[2]), float(si), float(c)]
            self._dirty.add(k)

    # --- incremental array management ---
    def ensure_array(self):
        if self._arr is not None:
            return
        self._keys = list(self.acc.keys())
        n = len(self._keys)
        self._arr = np.empty((max(1024, n), 4), np.float32)  # pre-allocate
        self._row = {}
        for i, k in enumerate(self._keys):
            self._row[k] = i
            a = self.acc[k]; c = max(1.0, a[4])
            self._arr[i, 0] = a[0] / c
            self._arr[i, 1] = a[1] / c
            self._arr[i, 2] = a[2] / c
            self._arr[i, 3] = a[3] / c
        self._dirty.clear()

    def apply_dirty(self):
        if self._arr is None:
            self.ensure_array()
            return
        for k in list(self._dirty):
            a = self.acc[k]; c = max(1.0, a[4])
            if k in self._row:
                i = self._row[k]
            else:
                i = len(self._keys)
                self._keys.append(k)
                self._row[k] = i
                if i >= len(self._arr):
                    grow = max(1024, int(0.25 * (i + 1)))
                    big = np.empty((len(self._arr) + grow, 4), np.float32)
                    big[:len(self._arr)] = self._arr
                    self._arr = big
            self._arr[i, 0] = a[0] / c
            self._arr[i, 1] = a[1] / c
            self._arr[i, 2] = a[2] / c
            self._arr[i, 3] = a[3] / c
            self._dirty.remove(k)

    def view_points(self) -> np.ndarray:
        if self._arr is None:
            return np.empty((0, 4), np.float32)
        n = len(self._keys)
        return self._arr[:n]

# --------- PipelineSLAM ---------
import queue, threading, time

class PipelineSLAM:
    """Three-stage SLAM pipeline: loader -> odometry/mapping -> loop/global-opt"""
    def __init__(
        self,
        files,
        preprocess_fn,
        map_voxel=0.45, icp_voxel=0.8,
        loop_gap=90, loop_thresh=0.35,
        optimize_every=200, enable_loop=True,
        loop_check_every=10, loop_topk=10, loop_min_icp_fit=0.35,
        sc_n_ring=20, sc_n_sector=60,
        window_horizon=0,
        kf_dist_m=2.5, kf_yaw_deg=12.0,
        max_queue=4,
    ):
        # config
        self.files = files
        self.preprocess = preprocess_fn
        self.icp_voxel = float(icp_voxel)
        self.enable_loop = bool(enable_loop)
        self.loop_gap = int(loop_gap)
        self.loop_thresh = float(loop_thresh)
        self.optimize_every = int(optimize_every)
        self.loop_check_every = int(loop_check_every)
        self.loop_topk = int(loop_topk)
        self.loop_min_icp_fit = float(loop_min_icp_fit)
        self.window_h = int(window_horizon)

        self.sc_n_ring = int(sc_n_ring)
        self.sc_n_sector = int(sc_n_sector)
        self.kf_dist_m = float(kf_dist_m)
        self.kf_yaw_deg = float(kf_yaw_deg)

        # shared state
        self.map = VoxelMap(map_voxel)
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        self.poses = [np.eye(4)]
        self.frames: list[np.ndarray] = []   # local (preprocessed) frames
        self.descs: list[Optional[np.ndarray]] = [None]
        self.progress = 0.0

        # keyframe DB (loop)
        self.kf_indices: list[int] = []
        self.kf_poses:   list[np.ndarray] = []
        self.kf_scs:     list[np.ndarray] = []
        self.kf_rkeys:   list[np.ndarray] = []
        self._last_kf_pose = np.eye(4, dtype=np.float32)

        # concurrency
        self.q_load = queue.Queue(maxsize=max_queue)
        self.q_odo  = queue.Queue(maxsize=max_queue)
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()

        # map publish handshake
        self._map_dirty = False
        self.map_version = 0

        # runtime flags
        self.min_d = 0.0; self.max_d = 0.0; self.ds_voxel = 0.35; self.loop_on = True

        # indices
        self.n = len(files)
        self.next_load_idx = 0
        self.last_processed_idx = -1

        # logging
        self._t0 = time.perf_counter()
        self._last_log_block = -1   # 0.. floor(k/500)

        self.map_lock = threading.Lock() 

        self._icp_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ICP")

    # --------------- life-cycle ---------------
    def start(self):
        self.stop_ev.clear()
        self.t_load = threading.Thread(target=self._worker_loader, daemon=True)
        self.t_odo  = threading.Thread(target=self._worker_odometry_map, daemon=True)
        self.t_loop = threading.Thread(target=self._worker_loop_global, daemon=True)
        self.t_load.start(); self.t_odo.start(); self.t_loop.start()

    def stop(self):
        self.stop_ev.set()
        # unblock queues
        try:
            self.q_load.put_nowait(None)
        except Exception:
            pass
        try:
            self.q_odo.put_nowait(None)
        except Exception:
            pass

    @property
    def running(self):  # for UI
        return not self.stop_ev.is_set() and (self.last_processed_idx + 1 < self.n)

    @property
    def idx(self):
        return self.last_processed_idx + 1

    # --------------- helpers ---------------
    def _mark_dirty(self):
        self._map_dirty = True
        self.map_version += 1

    def _needs_keyframe(self, T_k: np.ndarray) -> bool:
        # distance OR yaw change
        p_new = T_k[:3, 3]; p_old = self._last_kf_pose[:3, 3]
        dp = float(np.linalg.norm(p_new - p_old))
        if dp >= max(0.1, self.kf_dist_m):
            return True
        R_new = T_k[:3, :3]; R_old = self._last_kf_pose[:3, :3]
        dR = R_old.T @ R_new
        yaw = float(np.degrees(np.arctan2(dR[1,0], dR[0,0])))
        return abs(yaw) >= max(1.0, self.kf_yaw_deg)

    def _loopdb_add(self, k: int, sc: np.ndarray):
        self.kf_indices.append(k)
        self.kf_poses.append(self.poses[-1].copy())
        self.kf_scs.append(sc)
        self.kf_rkeys.append(sc_ring_key(sc))

    def _loopdb_query_topk(self, sc: np.ndarray, topk: int) -> list[int]:
        if not self.kf_rkeys:
            return []
        K = np.asarray(self.kf_rkeys, dtype=np.float32)
        q = sc_ring_key(sc).reshape(1, -1)
        num = (K @ q[0])
        den = np.linalg.norm(K, axis=1) * (np.linalg.norm(q))
        sim = num / (den + 1e-9)
        order = np.argsort(-sim)[: min(topk, K.shape[0])]
        return [self.kf_indices[i] for i in order]
    
    def icp_with_timeout(
        self,
        src_pts_np,
        tgt_pts_np,
        voxel,
        init,
        timeout=3.0,
        frame_idx=None,   # optional; for logging only
    ):
        """
        Runs icp_transform_backend in a tiny thread pool with a hard timeout.
        Accepts both positional and keyword arguments (frame_idx is optional).
        """
        def _job():
            return icp_transform_backend(src_pts_np, tgt_pts_np, voxel, init)

        fut = self._icp_pool.submit(_job)
        try:
            T, fit = fut.result(timeout=timeout)
            if frame_idx is not None:
                log(f"[ICP ] k={frame_idx} ok fit={fit:.3f}")
            return T, fit
        except Exception as e:
            # Ensure the worker is cancelled if still pending
            try:
                fut.cancel()
            except Exception:
                pass
            if frame_idx is not None:
                log(f"[ICP ] k={frame_idx} FAIL: {e!r}")
            raise


    # --------------- workers ---------------
    def _worker_loader(self):
        log("[LOAD] start")
        while not self.stop_ev.is_set() and self.next_load_idx < self.n:
            k = self.next_load_idx
            try:
                log(f"[LOAD] prep frame {k} ...")
                t0 = time.perf_counter()
                pts = self.preprocess(k, self.min_d, self.max_d, self.ds_voxel)  # cached
                dt = time.perf_counter() - t0

                while not self.stop_ev.is_set():
                    try:
                        self.q_load.put((k, pts), timeout=0.5)
                        log(f"[LOAD] pushed frame {k}  pts={len(pts)}  ({dt:.3f}s)", flush=True)
                        self.next_load_idx += 1
                        break
                    except queue.Full:
                        # Queue is full; do not recompute; just wait/retry
                        time.sleep(0.01)

            except Exception as e:
                print("[LOAD] fatal:", repr(e))
                import traceback; traceback.print_exc()
                self.next_load_idx += 1
                continue
        try:
            self.q_load.put(None, timeout=0.5)
        except Exception:
            pass
        log("[LOAD] EOS", flush=True)

    def _worker_odometry_map(self):
        log("[ODO ] start", flush=True)
        prev_pts = None
        try:
            while not self.stop_ev.is_set():
                try:
                    item = self.q_load.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    try: self.q_odo.put(None, timeout=0.5)
                    except Exception: pass
                    log("[ODO ] EOS", flush=True)
                    break

                k, pts_k = item
                log(f"[ODO ] got frame {k}  pts={len(pts_k)}", flush=True)

                if k == 0:
                    sc0 = make_scan_context(pts_k, n_ring=self.sc_n_ring, n_sector=self.sc_n_sector)
                    with self.lock:
                        self.frames.append(pts_k)
                        self.descs[0] = sc0
                        # (leave as-is) initial insert for k=0
                        self.map.insert(pts_k, np.eye(4))
                        self._mark_dirty()
                        self.last_processed_idx = 0
                        self.progress = 0.0 if self.n <= 1 else (0 / (self.n - 1))
                    prev_pts = pts_k
                    try: self.q_odo.put((k, pts_k[::4], np.eye(4, dtype=np.float32), True), timeout=0.5)
                    except queue.Full: pass
                    continue

                # ---- normal path (k>0) ----
                if pts_k is None or len(pts_k) == 0 or prev_pts is None or len(prev_pts) == 0:
                    log(f"[ODO ] skip ICP at k={k} (empty cloud)", flush=True)
                    prev_pts = pts_k
                    continue

                try:
                    T_cur_to_prev, fit = self.icp_with_timeout(
                        pts_k,
                        prev_pts if prev_pts is not None else pts_k,
                        self.icp_voxel,
                        np.eye(4, dtype=np.float32),
                        timeout=2.5,
                        frame_idx=k,
                    )
                except Exception as e:
                    print(f"[ODO ] ICP failed, using identity. err={e!r}")
                    T_cur_to_prev, fit = (np.eye(4, dtype=np.float32), 0.0)

                MIN_ODOM_FIT = 0.10
                with self.lock:
                    T_k = self.poses[-1] @ T_cur_to_prev
                    self.poses.append(T_k)
                    self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_k.copy()))

                    if fit >= MIN_ODOM_FIT:
                        info = np.identity(6, dtype=np.float32) * max(100.0 * fit, 10.0)
                        self.pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                k - 1, k, np.linalg.inv(T_cur_to_prev), info, uncertain=False
                            )
                        )
                    else:
                        info = np.identity(6, dtype=np.float32) * 1e-3
                        self.pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                k - 1, k, np.linalg.inv(T_cur_to_prev), info, uncertain=True
                            )
                        )

                # (unchanged) compute SC and keyframe logic
                sc_k = make_scan_context(pts_k, n_ring=self.sc_n_ring, n_sector=self.sc_n_sector)
                if self._needs_keyframe(T_k):
                    with self.lock:
                        self._loopdb_add(k, sc_k)
                        self._last_kf_pose = T_k.copy()

                # ---- single map insert (keep this one) ----
                with self.map_lock:
                    self.map.insert(pts_k, T_k)
                self._mark_dirty()  # mark once; remove any duplicate bumps

                with self.lock:
                    # (keep only bookkeeping here; removed the duplicate _map_dirty / map_version)
                    self.last_processed_idx = k
                    self.progress = k / max(1, self.n - 1)
                    blk = k // 500
                    if blk != self._last_log_block and k % 500 == 0:
                        now = time.perf_counter()
                        dt = now - self._t0
                        self._t0 = now
                        print(f"[PIPE] {k} frames processed (+500) in {dt:.2f}s  | avg {500.0/max(dt,1e-6):.2f} FPS", flush=True)
                        self._last_log_block = blk

                prev_pts = pts_k

                try: self.q_odo.put((k, pts_k[::4], T_k.astype(np.float32), False), timeout=0.5)
                except queue.Full: pass

        except Exception as e:
            import traceback
            print("[ODO ] fatal:", repr(e), flush=True)
            traceback.print_exc()


    def _worker_loop_global(self):
        log("[LOOP] start", flush=True)
        try:
            while not self.stop_ev.is_set():
                try:
                    item = self.q_odo.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    log("[LOOP] EOS", flush=True)
                    break

                k, pts_k_light, T_k, is_kf = item
                did_add_loop_edge = False

                # ---------- Bounded loop-closure ----------
                if (self.enable_loop and self.loop_on and
                    k > self.loop_gap and
                    (k % self.loop_check_every == 0) and
                    len(self.kf_indices) > 1 and
                    pts_k_light is not None and len(pts_k_light) > 0):

                    sc_k = make_scan_context(
                        pts_k_light, n_ring=self.sc_n_ring, n_sector=self.sc_n_sector
                    )
                    cand = self._loopdb_query_topk(sc_k, self.loop_topk)

                    best = (None, 1e9, 0)  # (j, dist, shift)
                    for j in cand:
                        if (k - j) < self.loop_gap:
                            continue
                        try:
                            sc_j = self.kf_scs[self.kf_indices.index(j)]
                        except ValueError:
                            continue
                        shift, dist = sc_best_shift_fft(sc_k, sc_j)
                        if dist < best[1]:
                            best = (j, dist, shift)

                    j, dist, shift = best
                    if j is not None and dist < self.loop_thresh:
                        yaw = 2.0 * np.pi * (shift / float(self.sc_n_sector))

                        # prepare data for verification (without holding the lock during ICP)
                        with self.lock:
                            if (j < len(self.frames) and j < len(self.poses) and
                                k < len(self.frames) and k < len(self.poses)):
                                init = (np.linalg.inv(self.poses[j]) @ self.poses[k]) @ rotz(yaw)
                                pts_j = self.frames[j]
                            else:
                                init, pts_j = None, None

                        # geometric verify
                        if (init is not None and pts_j is not None and
                            len(pts_j) > 0 and len(pts_k_light) > 0):
                            T_cur_to_j, fit_loop = icp_transform_backend(
                                pts_k_light, pts_j, voxel=0.8, init=init
                            )
                            if fit_loop >= self.loop_min_icp_fit:
                                with self.lock:
                                    info_l = (np.identity(6, dtype=np.float32) *
                                            max(50.0 * fit_loop, 10.0))
                                    self.pose_graph.edges.append(
                                        o3d.pipelines.registration.PoseGraphEdge(
                                            j, k, np.linalg.inv(T_cur_to_j),
                                            info_l, uncertain=True
                                        )
                                    )
                                    did_add_loop_edge = True

                # ---------- Decide whether to optimize now ----------
                # Optimize on cadence (every N) — that’s enough; no duplicate optimize in loop block.
                if (k % self.optimize_every) == 0 and k > 0:
                    # Only run if data up to k is actually available
                    with self.lock:
                        have_upto = min(self.last_processed_idx,
                                        len(self.frames) - 1,
                                        len(self.poses)  - 1)
                    if k <= have_upto:
                        self._run_global_opt_and_reintegrate(k)
                    # else: skip this cycle; odometry hasn’t caught up yet

        except Exception as e:
            import traceback
            print("[LOOP] fatal:", repr(e), flush=True)
            traceback.print_exc()

    def _run_global_opt_and_reintegrate(self, upto_k: int):
        with self.lock:
            # --- run global opt ---
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=max(2.0 * self.icp_voxel, 1.0),
                edge_prune_threshold=0.25,
                preference_loop_closure=2.0,
                reference_node=0,
            )
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option,
            )

            # --- update poses without truncating the tail ---
            optimized = [n.pose.copy() for n in self.pose_graph.nodes]
            if len(optimized) < len(self.poses):
                # keep the tail poses as-is so we can still render already-inserted frames
                optimized.extend(self.poses[len(optimized):])
            self.poses = optimized

            # --- decide reintegration span ---
            # Build up to what we actually processed, not upto_k
            built_max = min(
                self.last_processed_idx,           # last odometry-processed frame
                len(self.frames) - 1,
                len(self.poses)  - 1,
            )
            if built_max < 0:
                return

            if self.window_h > 0:
                start_i = max(0, built_max - self.window_h)
            else:
                start_i = 0  # full map when no sliding window

            # --- rebuild into a fresh VoxelMap, then swap ---
            new_map = VoxelMap(self.map.voxel)
            for i in range(start_i, built_max + 1):
                if i >= len(self.frames) or self.frames[i] is None:
                    continue
                pts_i = self.frames[i]
                Ti    = self.poses[i]
                if pts_i is not None and len(pts_i) and Ti is not None:
                    new_map.insert(pts_i, Ti)

            self.map = new_map
            self._map_dirty = True
            self.map_version += 1


def _slam_is_alive(obj) -> bool:
    # Works for Thread, Process, or plain classes
    if obj is None:
        return False
    if hasattr(obj, "is_alive"):
        try:
            return bool(obj.is_alive())
        except TypeError:
            return bool(obj.is_alive)   # some implementations expose a property
    # Fallback for our pipeline classes
    return bool(getattr(obj, "running", False))

def _slam_stop(obj):
    if obj is None:
        return
    # Prefer an explicit stop()
    if hasattr(obj, "stop"):
        try:
            obj.stop()
        except Exception:
            pass
    # If it's a Thread/Process, try to join briefly
    if hasattr(obj, "join"):
        try:
            obj.join(timeout=0.2)
        except Exception:
            pass

def _slam_start(obj):
    if obj is None:
        return
    if hasattr(obj, "start"):
        print("[PIPE] starting threads...")
        obj.start()
    else:
        # If your PipelineSLAM uses an internal thread, expose start()
        # If not, at least set .running = True and kick an internal loop
        setattr(obj, "running", True)

# --------- Camera helpers ---------
def fit_camera(renderer, margin=2.4):
    if renderer is None: return
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    try: cam.Zoom(1.0 / float(margin))
    except Exception: pass
    renderer.ResetCameraClippingRange()

_first_fit_done = {"left": False, "right": False}
def fit_camera_once(renderer, side: str, margin=2.4):
    if not _first_fit_done[side]:
        fit_camera(renderer, margin)
        _first_fit_done[side] = True

def extract_pose_vectors(T: np.ndarray):
    p = T[:3, 3].astype(float)
    R = T[:3, :3].astype(float)
    fwd = R @ np.array([1.0, 0.0, 0.0])
    up  = R @ np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    def norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    return p, norm(fwd), norm(left), norm(up)

_last_pose = {"T": np.eye(4, dtype=np.float32)}

def sc_ring_key(sc: np.ndarray) -> np.ndarray:
    return sc.max(axis=0).astype(np.float32, copy=False)

def sc_best_shift_fft(sc1: np.ndarray, sc2: np.ndarray) -> tuple[int, float]:
    a = sc1.mean(axis=0).astype(np.float32, copy=False)
    b = sc2.mean(axis=0).astype(np.float32, copy=False)
    fa = np.fft.rfft(a); fb = np.fft.rfft(b)
    xcorr = np.fft.irfft(fa * np.conjugate(fb), n=a.size)
    shift = int(np.argmax(xcorr))
    b2 = np.roll(sc2, shift, axis=1)
    num = (sc1 * b2).sum(axis=1)
    den = np.linalg.norm(sc1, axis=1) * np.linalg.norm(b2, axis=1) + 1e-9
    dist = 1.0 - float((num / den).mean())
    return shift, dist

def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return T

# --------- App ---------
def build_app(seq_dir: str, fps: int = 10):
    files = list_kitti_bins(seq_dir)

    @lru_cache(maxsize=256)
    def load_frame(i: int) -> np.ndarray:
        return load_kitti_bin(str(files[i]))

    @lru_cache(maxsize=1024)
    def preprocess_frame(idx: int, min_d: float, max_d: float, voxel: float) -> np.ndarray:
        t0 = time.perf_counter()
        pts = load_frame(idx)  # may be heavy I/O
        t1 = time.perf_counter()
        pts = crop_distance(pts, float(min_d), float(max_d))
        if voxel and float(voxel) > 0:
            pts = voxel_downsample(pts, float(voxel))
        t2 = time.perf_counter()
        if (idx % 100) == 0 or idx <= 2:  # keep it light
            log(f"[PP ] i={idx} load={t1-t0:.3f}s crop+ds={t2-t1:.3f}s out={len(pts)}")
        return pts

    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller

    def _on_client_connected(client_id=None, **_):
        client_connected["v"] = True
    def _on_client_exited(client_id=None, **_):
        client_connected["v"] = False
        # also stop any playback / timers you keep on the server side
        state.play = False

    try:
        server.on_client_connected.add(_on_client_connected)
    except Exception:
        pass
    try:
        server.on_client_exited.add(_on_client_exited)
    except Exception:
        pass

    # ----- Render window -----
    rw = vtk.vtkRenderWindow()
    try: rw.SetNumberOfLayers(2)
    except Exception: pass
    if IS_WINDOWS:
        try: rw.SetOffScreenRendering(0)
        except Exception: pass
    else:
        try: rw.SetOffScreenRendering(1)
        except Exception: pass
    try: rw.SetMultiSamples(0)
    except Exception: pass
    try: rw.SetAlphaBitPlanes(0)
    except Exception: pass

    ren_left  = vtk.vtkRenderer(); ren_left.SetBackground(0.07, 0.07, 0.10)
    ren_right = vtk.vtkRenderer(); ren_right.SetBackground(0.05, 0.05, 0.08)
    ren_marker = vtk.vtkRenderer()
    try: ren_marker.SetLayer(1)
    except Exception: pass
    try: ren_marker.SetBackgroundAlpha(0.0)
    except Exception: pass

    rw.AddRenderer(ren_left); rw.AddRenderer(ren_right); rw.AddRenderer(ren_marker)
    ren_left.SetViewport(0.0, 0.0, 0.5, 1.0)
    ren_right.SetViewport(0.5, 0.0, 1.0, 1.0)
    ren_marker.SetViewport(ren_right.GetViewport()); ren_marker.SetActiveCamera(ren_right.GetActiveCamera())

    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)
    try: iren.EnableRenderOff()
    except Exception: pass
    style_trackball = vtk.vtkInteractorStyleTrackballCamera()
    style_image     = vtk.vtkInteractorStyleImage()
    iren.SetInteractorStyle(style_image)

    def apply_interactor_style():
        if bool(state.pan_mode): iren.SetInteractorStyle(style_image)
        else: iren.SetInteractorStyle(style_trackball)
        try: iren.Initialize()
        except Exception: pass
        ctrl.view_update()
    ctrl.apply_interactor_style = apply_interactor_style

    # Pipelines
    map_mapper_fast = vtk.vtkPointGaussianMapper(); map_mapper_fast.EmissiveOn(); map_mapper_fast.SetScaleFactor(0.20)
    map_mapper_std  = vtk.vtkPolyDataMapper();      map_mapper_std.SetColorModeToDirectScalars()
    map_actor = vtk.vtkActor(); map_actor.SetMapper(map_mapper_std); map_actor.GetProperty().SetOpacity(0.72)
    ren_right.AddActor(map_actor)

    scan_mapper_fast = vtk.vtkPointGaussianMapper(); scan_mapper_fast.EmissiveOn(); scan_mapper_fast.SetScaleFactor(0.26)
    scan_mapper_std  = vtk.vtkPolyDataMapper();      scan_mapper_std.SetColorModeToDirectScalars()
    scan_actor = vtk.vtkActor(); scan_actor.SetMapper(scan_mapper_std)
    scan_prop = scan_actor.GetProperty(); scan_prop.SetRenderPointsAsSpheres(1)
    ren_left.AddActor(scan_actor)

    # Pose marker (overlay)
    marker_src = vtk.vtkConeSource(); marker_src.SetRadius(0.8); marker_src.SetHeight(2.4); marker_src.SetDirection(1, 0, 0); marker_src.SetResolution(32)
    marker_mapper = vtk.vtkPolyDataMapper(); marker_mapper.SetInputConnection(marker_src.GetOutputPort())
    marker_actor = vtk.vtkActor(); marker_actor.SetMapper(marker_mapper); marker_actor.SetPickable(False); marker_actor.SetVisibility(False)
    mp = marker_actor.GetProperty(); mp.SetColor(1.0, 0.2, 0.2); mp.SetLighting(False)
    try: mp.SetDepthTestOff()
    except Exception: pass
    ren_marker.AddActor(marker_actor)

    ren_left.ResetCamera(); ren_right.ResetCamera()

    # ---------- State (RT defaults) ----------
    state.trame__title = "KITTI LiDAR SLAM (RT-optimized)"
    state.seq_dir = str(Path(seq_dir).resolve())
    state.frame = 0
    state.n_frames = len(files); state.max_frame = max(0, state.n_frames - 1)

    # viz
    state.point_size_scan = 1
    state.point_size_map  = 1
    state.splat_size      = 0.20
    state.map_opacity = 0.72
    state.color_mode = "Turbo"
    state.auto_intensity = True
    state.i_low = 0.0; state.i_high = 1.0
    state.min_d = 0.0; state.max_d = 0.0
    state.show_current = False; state.use_world_pose = True
    state.highlight_scan = False
    state.map_stride = 8
    state.scan_stride = 3
    state.use_fast_mapper = True
    state.interacting = False

    # layout & interaction
    state.split_view = False
    state.map_only_on_build = True
    state.pan_mode = True

    # camera follow
    state.follow_pose = False
    state.cam_back = 30.0; state.cam_up = 15.0; state.cam_side = 0.0

    # Pose marker
    state.show_pose_marker = True
    state.marker_size = 2.5
    state.marker_auto_scale = True
    state.marker_scale_k = 0.035
    state.marker_size_min = 1.0

    # playback
    state.play = False; state.fps = int(fps)

    # SLAM controls (RT-friendly)
    state.slam_on = True
    state.loop_on = True
    state.icp_voxel = 0.5; state.ds_voxel = 0.35; state.map_voxel = 0.45
    state.loop_gap = 90; state.loop_thresh = 0.35; state.optimize_every = 200
    state.loop_stride = 6
    state.rebuild_after_opt = False
    state.window_horizon = 0          # 0=off; e.g., 300 keeps last 30s at 10Hz

    # progress
    state.map_building = False; state.map_progress = 0.0

    # Map publish performance caps
    state.max_publish_points = 300_000
    state.map_publish_every_n = 8

    # CPU⇄GPU toggle
    state.use_gpu = False
    # --- cadence / throttling ---
    state.ui_publish_stride = 10     # update visualization every N frames (5–10 is good)

    # --- Loop-closure (accurate & bounded) ---
    state.sc_n_ring = 20
    state.sc_n_sector = 60          # fewer sectors -> faster & robust; works well with FFT
    state.kf_dist_m = 2.5           # create a keyframe every ~4 m traveled
    state.kf_yaw_deg = 12.0         # or if yaw changed by >= 12 deg
    state.loop_check_every = 5     # attempt loop closure every N frames
    state.loop_topk = 10            # candidates fetched from ring-key index
    state.loop_thresh = 0.32        # SC distance after best shift (0.3-0.35 typical)
    state.loop_min_icp_fit = 0.35   # reject if geometric verify is weak
    state.loop_verify = "fgr"       # 'fgr' or 'icp'

    _iw_cache = {"lo": 0.0, "hi": 1.0}
    _last_viz_frame = {"f": -1}
    last_map_version = {"v": -1}
    last_published_version = {"v": -1}
    pushing = {"v": False}
    last_fast = {"v": False}
    _last_push = {"t": 0.0}
    MIN_PUSH_DT_IDLE = 1.0 / 2.0      # at most ~8 fps when idle
    MIN_PUSH_DT_INTER = 1.0 / 1.0     # at most ~4 fps while interacting (scroll/zoom)

    # Prefetch
    PREFETCH = 24
    pool = ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2))
    def prefetch(idx):
        end = min(idx + 1 + PREFETCH, state.n_frames)
        for j in range(idx + 1, end): pool.submit(load_frame, j)

    # ---- alpha/peeling off ----
    def configure_alpha_features(render_window, renderers):
        try: render_window.SetMultiSamples(0)
        except Exception: pass
        try: render_window.SetAlphaBitPlanes(0)
        except Exception: pass
        for r in renderers:
            try:
                r.SetUseDepthPeeling(0)
                r.SetMaximumNumberOfPeels(0)
                r.SetOcclusionRatio(0.0)
            except Exception: pass
    configure_alpha_features(rw, (ren_left, ren_right, ren_marker))

    _pre_lock = threading.RLock()

    # SLAM worker
    slam: Optional[PipelineSLAM] = None
    def make_slam() -> PipelineSLAM:
        def pp(idx, min_d, max_d, voxel):
            with _pre_lock:
                return preprocess_frame(idx, float(min_d), float(max_d), float(voxel))
        s = PipelineSLAM(
            files,
            preprocess_fn=pp,
            map_voxel=float(state.map_voxel),
            icp_voxel=float(state.icp_voxel),
            loop_gap=int(state.loop_gap),
            loop_thresh=float(state.loop_thresh),
            optimize_every=int(state.optimize_every),
            enable_loop=bool(state.loop_on),
            loop_check_every=int(state.loop_check_every),
            loop_topk=int(state.loop_topk),
            loop_min_icp_fit=float(state.loop_min_icp_fit),
            sc_n_ring=int(state.sc_n_ring),
            sc_n_sector=int(state.sc_n_sector),
            window_horizon=int(state.window_horizon),
            kf_dist_m=float(state.kf_dist_m),
            kf_yaw_deg=float(state.kf_yaw_deg),
        )
        s.min_d = float(state.min_d); s.max_d = float(state.max_d)
        s.ds_voxel = float(state.ds_voxel); s.loop_on = bool(state.loop_on)
        return s

    def push_view():
        # throttle same as before
        now = time.perf_counter()
        min_dt = MIN_PUSH_DT_INTER if bool(state.interacting) else MIN_PUSH_DT_IDLE
        if pushing["v"] or (now - _last_push["t"] < min_dt):
            return
        pushing["v"] = True
        try:
            # Always run the compute/publish path above push_view()
            # Only guard the actual websocket send:
            try:
                ctrl.view_update()           # this calls VtkRemoteView.update()
            except Exception as e:
                # socket may be gone; silence & keep pipeline running
                if "closing transport" in str(e).lower() or "websocket" in str(e).lower():
                    client_connected["v"] = False
                # don't re-raise
        finally:
            pushing["v"] = False
            _last_push["t"] = time.perf_counter()


    def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        if points.size == 0: return points
        P = points[:, :3]; Pw = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        out = np.zeros_like(points); out[:, :3] = Pw; out[:, 3] = points[:, 3] if points.shape[1] > 3 else 0.0
        return out

    # ---- GPU splats self-test ----
    def gpu_splats_self_test(poly_for_test=None) -> bool:
        """
        Verify PointGaussianMapper works on this driver, without permanently
        hijacking the mappers' inputs. We save/restore the previous inputs.
        """
        # Capture current inputs to restore after test
        prev_map_input = map_mapper_fast.GetInput()
        prev_scan_input = scan_mapper_fast.GetInput()
        try:
            if poly_for_test is None:
                test = np.array([[0, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 0, 1]], np.float32)
                poly_for_test = np_points_to_polydata(test, state.color_mode, (0.0, 1.0))
            map_mapper_fast.SetInputData(poly_for_test)
            scan_mapper_fast.SetInputData(poly_for_test)

            rw.Render()
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(rw)
            w2i.ReadFrontBufferOff()
            w2i.Update()
            img = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars())
            ok = bool(img is not None and img.size and img.max() > 0)
            return ok
        except Exception:
            return False
        finally:
            # Always restore the original inputs
            if prev_map_input is not None:
                map_mapper_fast.SetInputData(prev_map_input)
            if prev_scan_input is not None:
                scan_mapper_fast.SetInputData(prev_scan_input)

    # ---- Mapper swapping ----
    def swap_map_mapper():
        want_fast = bool(state.use_fast_mapper)
        is_fast   = isinstance(map_actor.GetMapper(), vtk.vtkPointGaussianMapper)
        if want_fast and not is_fast:
            map_actor.SetMapper(map_mapper_fast); map_actor.GetProperty().SetOpacity(1.0)
            map_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))
            if not gpu_splats_self_test(map_actor.GetMapper().GetInput()):
                map_actor.SetMapper(map_mapper_std); map_actor.GetProperty().SetOpacity(float(state.map_opacity))
                state.use_fast_mapper = False
        elif not want_fast and is_fast:
            map_actor.SetMapper(map_mapper_std); map_actor.GetProperty().SetOpacity(float(state.map_opacity))

    def swap_scan_mapper():
        want_fast = bool(state.use_fast_mapper)
        is_fast   = isinstance(scan_actor.GetMapper(), vtk.vtkPointGaussianMapper)
        if want_fast and not is_fast:
            scan_actor.SetMapper(scan_mapper_fast); scan_actor.GetProperty().SetOpacity(1.0)
            scan_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))
            if not gpu_splats_self_test(scan_actor.GetMapper().GetInput()):
                scan_actor.SetMapper(scan_mapper_std); scan_actor.GetProperty().SetOpacity(1.0)
                state.use_fast_mapper = False
        elif not want_fast and is_fast:
            scan_actor.SetMapper(scan_mapper_std); scan_actor.GetProperty().SetOpacity(1.0)

    def iw_for_render():
        # If interacting and auto, freeze the last window to avoid percentile work on every pan/zoom
        if state.auto_intensity and bool(state.interacting):
            return (_iw_cache["lo"], _iw_cache["hi"])
        if state.auto_intensity:
            # returning (1.0, 0.0) triggers auto-windowing inside np_points_to_polydata/publish path,
            # but we want to cache the result; so leave iw_for_render as a hint and set cache in publish.
            return (1.0, 0.0)
        return (float(state.i_low), float(state.i_high))


    def stride_for_map():
        base = max(1, as_int(state.map_stride, 1))
        return max(base, 6 if state.interacting else 1)

    def stride_for_scan():
        base = max(1, as_int(state.scan_stride, 1))
        return max(base, 2 if state.interacting else 1)

    # ---- Camera follow (right/map) ----
    def update_camera_follow_right():
        if not state.follow_pose: return
        if slam is None: return
        idx = int(state.frame)
        with slam.lock:
            if idx >= len(slam.poses): return
            T = slam.poses[idx].copy()
        p, fwd, left, up = extract_pose_vectors(T)
        pos = p - fwd * as_float(state.cam_back, 30.0) + up * as_float(state.cam_up, 15.0) + left * as_float(state.cam_side, 0.0)
        cam = ren_right.GetActiveCamera()
        cam.SetFocalPoint(float(p[0]), float(p[1]), float(p[2]))
        cam.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        cam.SetViewUp(float(up[0]), float(up[1]), float(up[2]))
        ren_right.ResetCameraClippingRange()

    # ---- Pose marker ----
    def update_pose_marker():
        if not bool(state.show_pose_marker):
            marker_actor.SetVisibility(False); return
        T = None
        try:
            if slam is not None:
                with slam.lock:
                    n = len(slam.poses)
                    if n > 0:
                        idx = min(int(state.frame), n - 1)
                        T = slam.poses[idx].copy()
        except Exception:
            T = None
        if T is None:
            T = _last_pose.get("T")
            if T is None:
                marker_actor.SetVisibility(False); return
        else:
            _last_pose["T"] = T

        p, fwd, left, up = extract_pose_vectors(T)
        R = np.column_stack((fwd, left, up))
        Tw = np.eye(4, dtype=np.float32); Tw[:3, :3] = R; Tw[:3, 3] = p
        marker_actor.SetUserMatrix(np_to_vtk_matrix(Tw))

        s = float(as_float(state.marker_size, 2.5))
        if bool(state.marker_auto_scale):
            cam = ren_right.GetActiveCamera()
            cpos = np.array(cam.GetPosition(), dtype=float)
            dist = max(1.0, float(np.linalg.norm(cpos - p)))
            s = max(float(as_float(state.marker_size_min, 1.0)),
                    min(float(as_float(state.marker_size, 2.5)),
                        float(as_float(state.marker_scale_k, 0.035)) * dist))
        marker_actor.SetScale(s, s, s)
        marker_actor.SetVisibility(True)

    # ---- View layout ----
    def update_view_layout():
        map_only = bool(state.map_only_on_build) and bool(state.map_building)
        split = bool(state.split_view) and not map_only
        if split:
            ren_left.SetViewport(0.0, 0.0, 0.5, 1.0)
            ren_right.SetViewport(0.5, 0.0, 1.0, 1.0)
        else:
            ren_right.SetViewport(0.0, 0.0, 1.0, 1.0)
            ren_left.SetViewport(0.0, 0.0, 0.0, 0.0)
        ren_marker.SetViewport(ren_right.GetViewport())
        ren_marker.SetActiveCamera(ren_right.GetActiveCamera())

    # ---- Persistent polydata for MAP ----
    map_poly = vtk.vtkPolyData()
    map_pts  = vtk.vtkPoints()
    map_rgb  = vtk.vtkUnsignedCharArray(); map_rgb.SetNumberOfComponents(3); map_rgb.SetName("RGB")
    map_verts = vtk.vtkCellArray()
    map_poly.SetPoints(map_pts)
    map_poly.GetPointData().SetScalars(map_rgb)
    map_poly.SetVerts(map_verts)
    map_mapper_std.SetInputData(map_poly)
    map_mapper_fast.SetInputData(map_poly)

    def publish_map_points(pts: np.ndarray):
        """Update persistent VTK arrays safely (VTK owns its memory)."""
        n = int(len(pts))

        # --- points (deep copy so VTK owns memory) ---
        vtk_xyz = numpy_to_vtk(pts[:, :3].astype(np.float32, copy=False), deep=True)
        map_pts.SetData(vtk_xyz)

        # --- verts (deep copy) ---
        # Build connectivity [1, id, 1, id, ...]
        ids = np.arange(n, dtype=np.int64)
        cells = np.empty((n, 2), dtype=np.int64)
        cells[:, 0] = 1
        cells[:, 1] = ids
        vtk_cells = numpy_to_vtkIdTypeArray(cells.ravel(order="C"), deep=True)
        map_verts.SetCells(n, vtk_cells)

        # --- colors (deep copy) ---
        lo, hi = iw_for_render()
        # after computing lo, hi for colors:
        _iw_cache["lo"] = float(lo)
        _iw_cache["hi"] = float(hi)

        if pts.shape[1] > 3:
            intens = pts[:, 3]
        else:
            intens = np.zeros(n, np.float32)

        if hi <= lo:
            lo = float(np.percentile(intens, 2.0)) if n else 0.0
            hi = float(np.percentile(intens, 98.0)) if n else 1.0
            if hi <= lo:
                hi = lo + 1.0

        t = np.clip((intens - lo) / max(1e-9, (hi - lo)), 0, 1)
        if state.color_mode == "Turbo":
            rgb_np = turbo_colormap(t)  # uint8 (n,3)
        else:
            g = (t * 255).astype(np.uint8)
            rgb_np = np.stack([g, g, g], axis=1)

        vtk_rgb = numpy_to_vtk(rgb_np.reshape(-1), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_rgb.SetNumberOfComponents(3)
        vtk_rgb.SetName("RGB")
        map_poly.GetPointData().SetScalars(vtk_rgb)

        # --- sizing ---
        if isinstance(map_actor.GetMapper(), vtk.vtkPointGaussianMapper):
            map_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))
        else:
            div = 2 if state.interacting else 1
            ps_map = max(1, as_int(state.point_size_map))
            map_actor.GetProperty().SetPointSize(max(1, ps_map // div))
            map_actor.GetProperty().SetOpacity(float(state.map_opacity))


    def update_scan_actor(idx: int):
        pts = preprocess_frame(idx, float(state.min_d), float(state.max_d), float(state.ds_voxel))
        if state.use_world_pose and slam is not None:
            with slam.lock:
                if idx < len(slam.poses): pts = transform_points(pts, slam.poses[idx])
        s = stride_for_scan()
        if s > 1 and len(pts) > 0: pts = pts[::s]
        const = (255, 240, 60) if state.highlight_scan else None
        poly = np_points_to_polydata(pts, state.color_mode, iw_for_render(), constant_rgb=const)
        scan_mapper_std.SetInputData(poly); scan_mapper_fast.SetInputData(poly)
        if isinstance(scan_actor.GetMapper(), vtk.vtkPointGaussianMapper):
            scan_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))
        else:
            div = 2 if state.interacting else 1
            ps_scan = max(1, as_int(state.point_size_scan))
            scan_prop.SetPointSize(max(1, ps_scan // div))
        swap_scan_mapper()
        if len(pts) > 0:
            fit_camera_once(ren_left, side="left", margin=2.2)

    def update_map_actor():
        nonlocal slam
        if slam is None:
            return

        # Read only small shared fields under slam.lock
        with slam.lock:
            ver = slam.map_version
            running = slam.running
            need_points = slam._map_dirty
            prog = float(slam.progress)

        state.map_progress = prog
        state.map_building = running

        if running and last_published_version["v"] >= 0:
            if (ver - last_published_version["v"]) < max(1, as_int(state.map_publish_every_n, 8)):
                return
        if ver == last_map_version["v"]:
            return
        if not need_points:
            return

        # Touch the VoxelMap under map_lock only
        slam.map.ensure_array()
        slam.map.apply_dirty()
        pts_full = slam.map.view_points()
        slam._map_dirty = False

        # Downsample for publishing and push to VTK (unchanged)
        s = stride_for_map()
        cap = max(150_000, as_int(state.max_publish_points, 300_000))
        pts = pts_full
        if len(pts) > cap:
            s = max(s, int(math.ceil(len(pts) / cap)))
        if s > 1 and len(pts) > 0:
            pts = pts[::s]

        publish_map_points(pts)
        swap_map_mapper()
        if len(pts) > 0:
            fit_camera_once(ren_right, side="right", margin=2.4)

        last_map_version["v"] = ver
        last_published_version["v"] = ver


    def update_frame(force=False):
        idx = int(state.frame)

        # Only publish when due: every ui_publish_stride frames or when interacting==True or on force
        publish_due = bool(force) or bool(state.interacting) or (
            idx != _last_viz_frame["f"] and (idx % max(1, as_int(state.ui_publish_stride, 10)) == 0)
        )

        if publish_due:
            _last_viz_frame["f"] = idx
            if state.show_current:
                update_scan_actor(idx)
            update_map_actor()              # map publish is already version-throttled inside
            update_view_layout()
            update_pose_marker()
            update_camera_follow_right()
            push_view()
        else:
            # Light path: follow camera only; no new map/scan uploads, no view push
            update_camera_follow_right()


    ctrl.update_frame = update_frame

    # periodic refresh
    @ctrl.trigger("refresh_map")
    def _refresh_map():
        if not state.interacting:
            update_map_actor()
            update_camera_follow_right()
            update_view_layout()
            update_pose_marker()
            if last_map_version["v"] >= 0: push_view()

    # playback
    @ctrl.trigger("advance")
    def _advance():
        if state.n_frames <= 0 or not state.play or state.interacting:
            return
        # avoid outrunning the builder
        if slam is not None and state.map_building:
            try:
                with slam.lock:
                    built_idx = int(slam.idx)
            except Exception:
                built_idx = int(state.frame)
            if int(state.frame) >= built_idx:
                return
        state.frame = (int(state.frame) + 1) % state.n_frames
        update_frame()   # publishes only when due


    @ctrl.trigger("fit_view")
    def _fit_view():
        fit_camera(ren_left, margin=2.2)
        fit_camera(ren_right, margin=2.4)
        ctrl.view_update()

    @ctrl.trigger("set_interacting")
    def _set_interacting(flag=0):
        prev = bool(state.interacting)
        state.interacting = bool(int(flag))
        if prev != state.interacting:
            update_frame()

    @ctrl.trigger("toggle_gpu")
    def _toggle_gpu():
        global USE_GPU
        state.use_gpu = bool(state.use_gpu and HAVE_CUPOCH)
        USE_GPU = bool(state.use_gpu)
        try: preprocess_frame.cache_clear()
        except Exception: pass
        nonlocal slam
        try:
            if _slam_is_alive(slam):
                _slam_stop(slam); time.sleep(0.05)
                if state.slam_on:
                    slam = make_slam(); slam.start()
        except Exception:
            pass
        print(f"[INFO] ICP backend: {'GPU (Cupoch)' if (USE_GPU and HAVE_CUPOCH) else 'CPU (Open3D)'}")

    def start_stop_slam():
        nonlocal slam
        if state.slam_on:
            if not _slam_is_alive(slam):
                slam = make_slam(); _slam_start(slam)
        else:
            if _slam_is_alive(slam):
                _slam_stop(slam)


    ctrl.start_stop_slam = start_stop_slam

    def build_full_map():
        """Start SLAM from scratch; RT-friendly config."""
        nonlocal slam, last_published_version
        # clear visible map immediately
        publish_map_points(np.empty((0, 4), np.float32)); ctrl.view_update()

        if _slam_is_alive(slam):
            _slam_stop(slam); time.sleep(0.05)

        n = len(files)
        state.optimize_every = 400
        state.loop_stride    = max(4, state.loop_stride)
        state.map_voxel      = max(0.5, as_float(state.map_voxel, 0.45))
        state.ds_voxel       = max(0.5, as_float(state.ds_voxel, 0.35))
        state.icp_voxel      = max(0.8,  as_float(state.icp_voxel, 0.8))
        state.sc_n_sector    = 60
        state.map_publish_every_n = 10
        state.ui_publish_stride = 12
        state.rebuild_after_opt = False

        slam = make_slam()
        state.map_building = True; state.map_progress = 0.0
        last_map_version["v"] = -1
        last_published_version["v"] = -1
        _first_fit_done["left"] = False; _first_fit_done["right"] = False
        
        _slam_start(slam)
        print("[UI] build_full_map: pipeline started")
        update_view_layout(); ctrl.view_update()
    ctrl.build_full_map = build_full_map

    def force_rebuild():
        nonlocal slam
        if slam is None:
            return
        with slam.map_lock:
            slam.map.ensure_array()
            slam.map.apply_dirty()
        with slam.lock:
            slam._map_dirty = True
            slam.map_version += 1
        _refresh_map()

    ctrl.force_rebuild = force_rebuild

    def reset_map():
        nonlocal slam
        # Stop threads if running
        if _slam_is_alive(slam):
            try:
                _slam_stop(slam)
                time.sleep(0.05)
            except Exception:
                pass

        # Best-effort clear internal structures if we still have an instance
        try:
            if slam is not None:
                with slam.lock:
                    try:
                        slam.map.clear()
                    except Exception:
                        pass
                    slam._map_dirty = False
        except Exception:
            pass

        # Drop the instance
        slam = None

        # Reset publish/version state
        last_map_version["v"] = -1
        last_published_version["v"] = -1
        _first_fit_done["left"] = False
        _first_fit_done["right"] = False
        state.map_progress = 0.0
        state.map_building = False

        # Clear VTK pipelines (map + scan) safely
        try:
            publish_map_points(np.empty((0, 4), np.float32))
        except Exception:
            pass

        try:
            empty_poly = np_points_to_polydata(np.empty((0, 4), np.float32), state.color_mode, (0.0, 1.0))
            scan_mapper_std.SetInputData(empty_poly)
            scan_mapper_fast.SetInputData(empty_poly)
        except Exception:
            pass

        # Reset actors/mappers/marker
        try:
            map_actor.SetMapper(map_mapper_std)
            scan_actor.SetMapper(scan_mapper_std)
        except Exception:
            pass
        try:
            marker_actor.SetVisibility(False)
        except Exception:
            pass

        # Refresh layout & view
        try:
            update_view_layout()
            ctrl.view_update()
        except Exception:
            pass

    ctrl.reset_map = reset_map

    def _on_client_exited(client_id=None, **_): state.play = False
    try: server.on_client_exited.add(_on_client_exited)
    except Exception: pass

    def _watchdog():
        while True:
            if slam is None or slam.stop_ev.is_set():
                break
            try:
                # Don’t take slam.lock (avoid deadlock)
                load_q = getattr(slam, "q_load", None)
                odo_q  = getattr(slam, "q_odo", None)

                print(
                    f"[WDOG] idx={slam.idx} "
                    f"progress={slam.progress:.2%} "
                    f"loadQ={load_q.qsize() if load_q else '?'} "
                    f"odoQ={odo_q.qsize() if odo_q else '?'} "
                    f"running={slam.running}"
                )
            except Exception as e:
                print(f"[WDOG] error: {e}")
            time.sleep(2.0)   # every 2 seconds

    # ---------- UI ----------
    with SinglePageLayout(server) as layout:
        layout.title.set_text("KITTI LiDAR SLAM (RT-optimized)")
        with layout.content:
            with h.Div(style="display:flex; gap:12px; align-items:flex-start; padding:12px;"):
                with h.Div(style="width:640px; min-width:320px;"):
                    h.Div(children=["Sequence: ", h.Code(children=[state.seq_dir])])
                    h.Hr()
                    h.Label("Frame")
                    h.Input(type="range", v_model=("frame", 0), min=0, max=("max_frame", 0), step=1, change=ctrl.update_frame, style="width:100%")
                    with h.Div(style="display:flex; gap:8px; margin-top:6px; align-items:center;"):
                        h.Button("Prev", click="frame=Math.max(0,frame-1); $event")
                        h.Button(children=["{{ play ? '❚❚' : '▶' }}"], click=(
                            "play=!play; "
                            "if(play){ if(window._t) clearInterval(window._t); "
                            "window._t=setInterval(()=>trigger('advance'), Math.max(10, 1000/Math.max(1, fps||10))); } "
                            "else { if(window._t){ clearInterval(window._t); window._t=null; } } $event"
                        ))
                        h.Button("Next", click="frame=Math.min(n_frames-1, frame+1); $event")
                        h.Span(children=["{{ frame + 1 }} / {{ n_frames }}"])
                    h.Button("Fit both views", click="trigger('fit_view')", style="width:100%; margin-top:6px;")
                    h.Div(style="font-size:12px; opacity:0.8; margin-top:4px;", children=[
                        h.Span("Pan = LEFT drag (Pan Mode ON). "),
                        h.Span("Toggle Pan Mode off to rotate with left-drag (Shift+Left pans).")
                    ])
                    h.Hr()

                    with h.Div(style="display:flex; gap:10px; align-items:center;"):
                        h.Input(type="checkbox", v_model=("split_view", True), change=ctrl.update_frame); h.Label("Split view (Scan | Map)")
                    with h.Div(style="display:flex; gap:10px; align-items:center;"):
                        h.Input(type="checkbox", v_model=("map_only_on_build", False), change=ctrl.update_frame); h.Label("Map-only during Build Full Map")
                    with h.Div(style="display:flex; gap:10px; align-items:center; margin-top:6px;"):
                        h.Input(type="checkbox", v_model=("pan_mode", True), change=ctrl.apply_interactor_style); h.Label("Pan Mode (left-drag pans)")

                    with h.Div(style="display:flex; gap:8px; align-items:center; margin-top:6px;"):
                        h.Input(type="checkbox", v_model=("use_fast_mapper", True), change=ctrl.update_frame)
                        h.Label("Use Fast GPU splats (turn OFF if artifacts)")
                    with h.Div(style="display:flex; gap:10px; align-items:center; margin-top:4px;"):
                        h.Label("Gaussian splat size")
                        h.Input(type="range", min=0.05, max=1.2, step=0.01, v_model=("splat_size", 0.20), change=ctrl.update_frame, style="flex:1")

                    with h.Div(style="display:flex; gap:12px; align-items:center; margin-top:6px;"):
                        h.Input(type="checkbox", v_model=("auto_intensity", True), change=ctrl.update_frame); h.Label("Auto intensity window")
                    with h.Div(style="display:flex; gap:8px;"):
                        h.Label("i_low");  h.Input(type="number", step="0.01", v_model=("i_low", 0.0),  change=ctrl.update_frame, style="width:40%")
                        h.Label("i_high"); h.Input(type="number", step="0.01", v_model=("i_high", 1.0), change=ctrl.update_frame, style="width:40%")

                    h.Hr()
                    with h.Div(style="display:flex; gap:10px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("follow_pose", False), change=ctrl.update_frame); h.Label("Follow current pose (map/right)")
                    with h.Div(style="display:flex; gap:10px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("show_pose_marker", True), change=ctrl.update_frame); h.Label("Show pose marker")
                    with h.Div(style="display:flex; gap:10px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("marker_auto_scale", True), change=ctrl.update_frame); h.Label("Marker auto-scale by camera distance")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Label("Marker size (cap)"); h.Input(type="range", min=0.3, max=15.0, step=0.1, v_model=("marker_size", 2.5), change=ctrl.update_frame, style="flex:1")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Label("Auto min"); h.Input(type="number", step="0.1", v_model=("marker_size_min", 1.0), change=ctrl.update_frame, style="width:30%")
                        h.Label("Auto scale k"); h.Input(type="number", step="0.005", v_model=("marker_scale_k", 0.035), change=ctrl.update_frame, style="width:40%")

                    h.Hr()
                    h.Label("Map render stride (base LOD)")
                    h.Input(type="range", min=1, max=10, step=1, v_model=("map_stride", 8), change=ctrl.update_frame, style="width:100%")
                    h.Label("Scan render stride (base LOD)")
                    h.Input(type="range", min=1, max=5, step=1, v_model=("scan_stride", 3), change=ctrl.update_frame, style="width:100%")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Label("Map pt size"); h.Input(type="range", min=1, max=8, step=1, v_model=("point_size_map", 1), change=ctrl.update_frame, style="flex:1")
                        h.Label("Opacity"); h.Input(type="range", min=0.2, max=1.0, step=0.05, v_model=("map_opacity", 0.72), change=ctrl.update_frame, style="flex:1")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Label("Scan pt size"); h.Input(type="range", min=1, max=10, step=1, v_model=("point_size_scan", 1), change=ctrl.update_frame, style="flex:1")
                        h.Input(type="checkbox", v_model=("highlight_scan", False), change=ctrl.update_frame); h.Label("Highlight current scan")

                    h.Hr()
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("show_current", True), change=ctrl.update_frame); h.Label("Show current scan (left)")
                        h.Input(type="checkbox", v_model=("use_world_pose", True), change=ctrl.update_frame); h.Label("Use SLAM pose for scan")

                    h.Hr()
                    h.Label("Distance crop [min, max] m (0 disables)")
                    with h.Div(style="display:flex; gap:8px;"):
                        h.Input(type="number", step="1", v_model=("min_d", 0.0), change=ctrl.update_frame, style="width:50%")
                        h.Input(type="number", step="1", v_model=("max_d", 0.0), change=ctrl.update_frame, style="width:50%")
                    h.Label("Downsample voxel (m) for current scan & SLAM (CPU)")
                    h.Input(type="range", min=0, max=0.8, step=0.02, v_model=("ds_voxel", 0.35), change=ctrl.update_frame, style="width:100%")

                    h.Hr()
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("use_gpu", HAVE_CUPOCH), change="trigger('toggle_gpu')")
                        h.Label(f"Use GPU (Cupoch) for ICP{' (not available)' if not HAVE_CUPOCH else ''}")

                    h.Hr()
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("slam_on", True), change=ctrl.start_stop_slam); h.Label("Run SLAM (live)")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("loop_on", True)); h.Label("Loop closure enabled")
                    h.Label("Loop gap"); h.Input(type="number", min=10, step=1, v_model=("loop_gap", 90))
                    h.Label("Loop threshold"); h.Input(type="number", step="0.01", v_model=("loop_thresh", 0.35))
                    h.Label("Optimize every N frames"); h.Input(type="number", min=5, step=1, v_model=("optimize_every", 200))
                    h.Label("Loop stride (candidate step)"); h.Input(type="number", min=1, step=1, v_model=("loop_stride", 6))
                    h.Label("Map voxel (m)"); h.Input(type="range", min=0.05, max=1.0, step=0.05, v_model=("map_voxel", 0.45))
                    h.Label("ICP voxel (m)"); h.Input(type="range", min=0.1, max=1.5, step=0.05, v_model=("icp_voxel", 0.8))
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("rebuild_after_opt", False)); h.Label("Rebuild map after optimization")
                    with h.Div(style="display:flex; gap:8px; align_items:center; margin-top:6px;"):
                        h.Label("Sliding window H (frames, 0=off)"); h.Input(type="number", min=0, step=10, v_model=("window_horizon", 0))

                    h.Hr()
                    h.Label("Map publish caps")
                    with h.Div(style="display:flex; gap:8px; align-items:center;"):
                        h.Label("Max points to publish"); h.Input(type="number", min=100000, step=50000, v_model=("max_publish_points", 300_000), change=ctrl.update_frame, style="width:40%")
                        h.Label("Publish every N versions"); h.Input(type="number", min=1, step=1, v_model=("map_publish_every_n", 8), change=ctrl.update_frame, style="width:30%")

                    with h.Div(style="display:flex; gap:8px; margin-top:8px;"):
                        h.Button("Build Full Map (run to end)", click=ctrl.build_full_map, style="flex:2")
                        h.Button("Force Rebuild Now", click=ctrl.force_rebuild, style="flex:1")
                        h.Button("Reset Map/SLAM", click=ctrl.reset_map, style="flex:1")

                    with h.Div(style="margin-top:6px; font-size:12px; opacity:0.8;"):
                        h.Div(children=["Map: ", h.Span(children=["{{ (map_progress*100).toFixed(1) }}%"]), "  ", h.Span(children=["{{ map_building ? '(running)' : '' }}"])])
                        h.Script("""
                        (function(){
                          const root = document.querySelector('.vtk-wrap');
                          if (!root || window._pauseHooksSet) return;
                          window._pauseHooksSet = true;
                          function stopTimer(){ if (window._mapTimer){ clearInterval(window._mapTimer); window._mapTimer = null; } }
                          function startTimer(){ if (!window._mapTimer){ window._mapTimer = setInterval(()=>{ try{ trigger('refresh_map'); }catch(e){} }, 900); } }
                          let resumeT = null, interT = null;
                          function markInteractingKick(){
                            if (!window._interacting){
                              window._interacting = true;
                              try { trigger('set_interacting', 1); } catch(e){}
                            }
                            if (interT) clearTimeout(interT);
                            interT = setTimeout(()=>{ window._interacting = false; try { trigger('set_interacting', 0); } catch(e){} }, 250);
                          }
                          const pauseThenResume = () => {
                            stopTimer();
                            markInteractingKick();
                            if (resumeT) clearTimeout(resumeT);
                            resumeT = setTimeout(startTimer, 260);
                          };
                          root.addEventListener('mousedown', pauseThenResume, {passive:true});
                          root.addEventListener('mousemove', pauseThenResume, {passive:true});
                          root.addEventListener('mouseup',   pauseThenResume, {passive:true});
                          root.addEventListener('wheel',     pauseThenResume, {passive:true});
                          startTimer();
                          window.addEventListener('beforeunload', ()=>{ stopTimer(); if (window._t) { clearInterval(window._t); window._t=null; } });
                        })();
                        """)

                with h.Div(style="flex:1 1 auto; min-width:520px; height: calc(100vh - 140px);"):
                    with vtkw.VtkRemoteView(
                        rw,
                        ref="view",
                        classes="vtk-wrap",
                        style="width:100%; height:100%;",
                        interactive_ratio=6,
                        interactive_quality=20,
                        still_ratio=1,
                        still_quality=45,
                        max_image_size=480,
                    ) as view:
                        def _safe_update():
                            try: view.update()
                            except Exception: return
                        ctrl.view_update = _safe_update
                        ctrl.update_frame()
                        ctrl.apply_interactor_style()

    if state.slam_on:
        try: ctrl.start_stop_slam()
        except Exception: pass

    # threading.Thread(target=_watchdog, daemon=True).start()

    return server

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default=r"D:\Work\Lidar_Map_Trame\dataset\2011_10_03\2011_10_03_drive_0027_sync\velodyne_points\data")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    server = build_app(args.seq_dir, fps=args.fps)
    server.start(port=args.port)
