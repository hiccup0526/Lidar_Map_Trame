from __future__ import annotations
"""
KITTI LiDAR viewer + simple SLAM (ICP + scan-context)
- Split view: left (current scan), right (global map)
- Persistent, always-on-top pose marker (overlay renderer)
- CPU⇄GPU toggle for ICP (GPU via Cupoch if available)
- Fast GPU splats path (PointGaussianMapper) with robust fallback
- Performance caps for map publishing
- Pan Mode ON by default => left-drag pans (no rotate)

Requested defaults:
- Use Fast GPU splats = ON
- Run SLAM = ON
- Follow current pose = OFF
- Highlight current scan = OFF
- Map pt size = 1, Scan pt size = 1
- Map stride = 8, Scan stride = 3
"""
import argparse, platform, threading, time, multiprocessing, os, math
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional

# Use all CPU cores for CPU-side ops (set BEFORE heavy libs)
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

IS_WINDOWS = platform.system() == "Windows"

# --------- VTK <-> NumPy helpers ---------
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
except Exception:
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy  # type: ignore

# --------- small coercion utils (fix UI strings) ---------
def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

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
    """Fast CPU voxel grid that preserves intensity via mean per cell."""
    if voxel is None or voxel <= 0:
        return points
    xyz = points[:, :3].astype(np.float32, copy=False)
    ijk = np.floor(xyz / float(voxel)).astype(np.int64)
    try:
        _, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    except TypeError:  # numpy<1.24
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
    if n == 0:
        poly.SetPoints(vtk.vtkPoints())
        return poly

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(points[:, :3].copy(), deep=True))
    poly.SetPoints(pts)

    # verts
    ids = np.arange(n, dtype=np.int64)
    cells = np.empty((n, 2), dtype=np.int64)
    cells[:, 0] = 1
    cells[:, 1] = ids
    ca = numpy_to_vtkIdTypeArray(cells.ravel(order="C"), deep=True)
    verts = vtk.vtkCellArray()
    verts.SetCells(n, ca)
    poly.SetVerts(verts)

    # colors
    if constant_rgb is not None:
        colors = np.tile(np.asarray(constant_rgb, dtype=np.uint8), (n, 1))
    else:
        intens = points[:, 3] if points.shape[1] > 3 else np.zeros(n, dtype=np.float32)
        lo, hi = i_win
        if hi <= lo:  # auto-window
            lo = float(np.percentile(intens, 2.0)) if intens.size else 0.0
            hi = float(np.percentile(intens, 98.0)) if intens.size else 1.0
            if hi <= lo: hi = lo + 1.0
        t = np.clip((intens - lo) / (hi - lo), 0, 1)
        if color_mode == "Turbo":
            colors = turbo_colormap(t)
        else:
            g = (t * 255).astype(np.uint8)
            colors = np.stack([g, g, g], axis=1)
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

# --------- ICP backend (GPU via Cupoch when enabled; else CPU via Open3D) ---------
USE_GPU = HAVE_CUPOCH  # runtime switch (also exposed in UI)

def icp_transform_backend(src_pts_np: np.ndarray, tgt_pts_np: np.ndarray, voxel: float, init=np.eye(4)):
    """ICP using Cupoch on GPU when enabled, else Open3D on CPU."""
    if USE_GPU and HAVE_CUPOCH:
        src = cph.geometry.PointCloud()
        src.points = cph.utility.Vector3fVector(src_pts_np[:, :3].astype(np.float32, copy=False))
        tgt = cph.geometry.PointCloud()
        tgt.points = cph.utility.Vector3fVector(tgt_pts_np[:, :3].astype(np.float32, copy=False))
        if voxel and voxel > 0:
            src = src.voxel_down_sample(float(voxel))
            tgt = tgt.voxel_down_sample(float(voxel))
        max_corr = max(2.0 * float(voxel), 1.0) if voxel and voxel > 0 else 1.0
        result = cph.registration.registration_icp(
            src, tgt, max_corr, init.astype(np.float32),
            cph.registration.TransformationEstimationPointToPoint(),
            cph.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        T = np.array(result.transformation, dtype=np.float32)
        fit = float(getattr(result, "fitness", 1.0))
        return T, fit

    # --- CPU fallback (Open3D) ---
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(src_pts_np[:, :3].astype(np.float64, copy=False))
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(tgt_pts_np[:, :3].astype(np.float64, copy=False))
    if voxel and voxel > 0:
        src = src.voxel_down_sample(float(voxel))
        tgt = tgt.voxel_down_sample(float(voxel))
    max_corr = max(2.0 * float(voxel), 1.0) if voxel and voxel > 0 else 1.0
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return np.array(result.transformation, dtype=np.float32), float(result.fitness)

# --------- SLAM: scan-context (simple) ---------
def make_scan_context(points: np.ndarray, n_ring=20, n_sector=60, max_range=80.0) -> np.ndarray:
    xyz = points[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x*x + y*y)
    a = np.arctan2(y, x)
    a = (a + np.pi) / (2 * np.pi)
    ring_idx = np.clip((r / max_range * n_ring).astype(int), 0, n_ring - 1)
    sector_idx = np.clip((a * n_sector).astype(int), 0, n_sector - 1)
    sc = np.full((n_ring, n_sector), -np.inf, dtype=np.float32)
    for i in range(len(z)):
        ri = ring_idx[i]; si = sector_idx[i]
        if z[i] > sc[ri, si]: sc[ri, si] = z[i]
    sc[sc == -np.inf] = 0.0
    row_max = sc.max(axis=1, keepdims=True) + 1e-6
    return sc / row_max

def scan_context_distance(sc1: np.ndarray, sc2: np.ndarray) -> float:
    n_ring, n_sector = sc1.shape
    best = 1e9
    for shift in range(n_sector):
        sc2s = np.roll(sc2, shift, axis=1)
        num = (sc1 * sc2s).sum(axis=1)
        den = np.linalg.norm(sc1, axis=1) * np.linalg.norm(sc2s, axis=1) + 1e-9
        sim = (num / den).mean()
        dist = 1.0 - sim
        if dist < best: best = float(dist)
    return best

# --------- Optimized VoxelMap ---------
class VoxelMap:
    def __init__(self, voxel=0.3):
        self.voxel = float(voxel)
        self.acc: dict[int, list[float]] = {}

    def clear(self):
        self.acc.clear()

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

    def to_points(self) -> np.ndarray:
        if not self.acc:
            return np.empty((0, 4), np.float32)
        out = np.empty((len(self.acc), 4), np.float32)
        for i, a in enumerate(self.acc.values()):
            c = a[4]
            out[i, 0] = a[0] / c
            out[i, 1] = a[1] / c
            out[i, 2] = a[2] / c
            out[i, 3] = a[3] / c
        return out

class SimpleSLAM(threading.Thread):
    """Pose graph SLAM with voxel map; publishes versioned downsampled map."""
    def __init__(self, files, preprocess_fn,
                 map_voxel=0.3, icp_voxel=0.4, loop_gap=30, loop_thresh=0.25,
                 optimize_every=30, enable_loop=True, rebuild_after_opt=True,
                 loop_stride=1):
        super().__init__(daemon=True)
        self.files = files
        self.preprocess = preprocess_fn
        self.icp_voxel = float(icp_voxel)
        self.loop_gap = int(loop_gap)
        self.loop_thresh = float(loop_thresh)
        self.optimize_every = int(optimize_every)
        self.enable_loop = bool(enable_loop)
        self.rebuild_after_opt = bool(rebuild_after_opt)
        self.loop_stride = max(1, int(loop_stride))

        self.map = VoxelMap(map_voxel)
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        self.poses = [np.eye(4)]
        self.descs = [None]
        self.frames: list[np.ndarray] = []

        self.idx = 0
        self.running = False
        self.lock = threading.Lock()
        self.latest_map = np.empty((0, 4), np.float32)
        self.progress = 0.0
        self.map_version = 0

        # live params
        self.min_d = 0.0; self.max_d = 0.0; self.ds_voxel = 0.2; self.loop_on = True

    def run(self):
        self.running = True
        n = len(self.files)
        while self.running and self.idx < n:
            try:
                self.process_one(self.idx)
                self.idx += 1
                self.progress = self.idx / max(1, n - 1)
            except Exception:
                self.idx += 1
        self.running = False

    def stop(self): self.running = False

    def rebuild_map(self, upto: Optional[int] = None):
        kmax = len(self.frames) if upto is None else min(upto + 1, len(self.frames))
        self.map.clear()
        for i in range(kmax):
            self.map.insert(self.frames[i], self.poses[i])
        with self.lock:
            self.latest_map = self.map.to_points()
            self.map_version += 1

    def process_one(self, k: int):
        pts_k = self.preprocess(k, self.min_d, self.max_d, self.ds_voxel)
        if k == len(self.frames): self.frames.append(pts_k)
        else: self.frames[k] = pts_k

        if k == 0:
            self.descs[0] = make_scan_context(pts_k)
            self.map.insert(pts_k, np.eye(4))
            with self.lock:
                self.latest_map = self.map.to_points()
                self.map_version += 1
            return

        # Align CURRENT -> PREVIOUS
        T_cur_to_prev, fit = icp_transform_backend(pts_k, self.frames[k - 1], self.icp_voxel, np.eye(4))
        T_k = self.poses[-1] @ T_cur_to_prev
        self.poses.append(T_k)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_k.copy()))
        info = np.identity(6) * max(100.0 * fit, 10.0)
        self.pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                k - 1, k, np.linalg.inv(T_cur_to_prev), info, uncertain=False
            )
        )

        # Loop closure (scan-context) with stride
        self.descs.append(make_scan_context(pts_k))
        if self.enable_loop and self.loop_on and k > self.loop_gap:
            best_j, best_d = -1, 1e9
            sc_k = self.descs[k]
            step = max(1, int(self.loop_stride))
            for j in range(0, k - self.loop_gap, step):
                d = scan_context_distance(sc_k, self.descs[j])
                if d < best_d: best_d, best_j = d, j
            if best_d < self.loop_thresh and best_j >= 0:
                init = np.linalg.inv(self.poses[best_j]) @ self.poses[k]
                T_cur_to_j, fit_loop = icp_transform_backend(pts_k, self.frames[best_j], self.icp_voxel, init)
                info_l = np.identity(6) * max(50.0 * fit_loop, 10.0)
                self.pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        best_j, k, np.linalg.inv(T_cur_to_j), info_l, uncertain=True
                    )
                )

        did_opt = False
        if k % self.optimize_every == 0 and k > 0:
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=max(2.0 * self.icp_voxel, 1.0),
                edge_prune_threshold=0.25,
                preference_loop_closure=1.0,
                reference_node=0,
            )
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option,
            )
            self.poses = [n.pose.copy() for n in self.pose_graph.nodes]
            did_opt = True

        if did_opt and self.rebuild_after_opt:
            self.rebuild_map(k)
        else:
            self.map.insert(pts_k, self.poses[-1])
            with self.lock:
                self.latest_map = self.map.to_points()
                self.map_version += 1

# --------- Camera helpers ---------
def fit_camera(renderer, margin=2.4):
    if renderer is None:
        return
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    try:
        cam.Zoom(1.0 / float(margin))
    except Exception:
        pass
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

# Cache last known pose for persistent marker
_last_pose = {"T": np.eye(4, dtype=np.float32)}

# --------- App ---------
def build_app(seq_dir: str, fps: int = 10):
    files = list_kitti_bins(seq_dir)

    @lru_cache(maxsize=256)
    def load_frame(i: int) -> np.ndarray:
        return load_kitti_bin(str(files[i]))

    @lru_cache(maxsize=1024)
    def preprocess_frame(idx: int, min_d: float, max_d: float, voxel: float) -> np.ndarray:
        pts = load_frame(idx)
        pts = crop_distance(pts, float(min_d), float(max_d))
        if voxel and float(voxel) > 0:
            pts = voxel_downsample(pts, float(voxel))  # keep CPU here to preserve intensity
        return pts

    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller

    # One RenderWindow, TWO base renderers + 1 overlay for marker
    rw = vtk.vtkRenderWindow()
    try: rw.SetNumberOfLayers(2)
    except Exception: pass
    if not IS_WINDOWS:
        try: rw.SetOffScreenRendering(1)
        except Exception: pass
    try: rw.SetShowWindow(False)
    except Exception: pass
    try: rw.SetMultiSamples(0)
    except Exception: pass

    ren_left  = vtk.vtkRenderer(); ren_left.SetBackground(0.07, 0.07, 0.10)
    ren_right = vtk.vtkRenderer(); ren_right.SetBackground(0.05, 0.05, 0.08)
    # Overlay renderer for marker (always on top)
    ren_marker = vtk.vtkRenderer()
    try: ren_marker.SetLayer(1)
    except Exception: pass
    try: ren_marker.SetBackgroundAlpha(0.0)
    except Exception: pass

    rw.AddRenderer(ren_left)
    rw.AddRenderer(ren_right)
    rw.AddRenderer(ren_marker)

    ren_left.SetViewport(0.0, 0.0, 0.5, 1.0)
    ren_right.SetViewport(0.5, 0.0, 1.0, 1.0)
    ren_marker.SetViewport(ren_right.GetViewport())
    ren_marker.SetActiveCamera(ren_right.GetActiveCamera())

    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)
    try: iren.EnableRenderOff()
    except Exception: pass
    style_trackball = vtk.vtkInteractorStyleTrackballCamera()
    style_image     = vtk.vtkInteractorStyleImage()   # left-drag pan
    iren.SetInteractorStyle(style_image)              # Pan Mode default ON

    def apply_interactor_style():
        if bool(state.pan_mode):
            iren.SetInteractorStyle(style_image)      # left-drag pans
        else:
            iren.SetInteractorStyle(style_trackball)  # left-drag rotates (Shift+Left pan)
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

    # ---- Pose marker (overlay) ----
    marker_src = vtk.vtkConeSource()
    marker_src.SetRadius(0.6)
    marker_src.SetHeight(1.8)
    marker_src.SetDirection(1, 0, 0)
    marker_src.SetResolution(32)

    marker_mapper = vtk.vtkPolyDataMapper()
    marker_mapper.SetInputConnection(marker_src.GetOutputPort())

    marker_actor = vtk.vtkActor()
    marker_actor.SetMapper(marker_mapper)
    marker_actor.SetPickable(False)
    marker_actor.SetVisibility(False)

    marker_prop = marker_actor.GetProperty()
    marker_prop.SetColor(1.0, 0.2, 0.2)
    marker_prop.SetLighting(False)
    try: marker_prop.SetDepthTestOff()
    except Exception: pass

    ren_marker.AddActor(marker_actor)

    ren_left.ResetCamera(); ren_right.ResetCamera()

    # ---------- State (requested defaults) ----------
    state.trame__title = "KITTI LiDAR SLAM (CPU⇄GPU ICP; Split View; Publish Caps)"
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
    state.show_current = True; state.use_world_pose = True
    state.highlight_scan = False
    state.map_stride = 8
    state.scan_stride = 3
    state.use_fast_mapper = True
    state.interacting = False

    # layout & interaction
    state.split_view = True
    state.map_only_on_build = True
    state.pan_mode = True        # Pan Mode ON => left-drag pans by default

    # camera follow (right/map)
    state.follow_pose = False
    state.cam_back = 30.0
    state.cam_up   = 15.0
    state.cam_side = 0.0

    # Pose marker
    state.show_pose_marker = True
    state.marker_size = 1.0
    state.marker_auto_scale = True
    state.marker_scale_k = 0.03
    state.marker_size_min = 0.6

    # playback
    state.play = False; state.fps = int(fps)

    # SLAM controls
    state.slam_on = True
    state.loop_on = True
    state.icp_voxel = 0.4; state.ds_voxel = 0.2; state.map_voxel = 0.3
    state.loop_gap = 30; state.loop_thresh = 0.25; state.optimize_every = 30
    state.loop_stride = 1
    state.rebuild_after_opt = True

    # progress
    state.map_building = False; state.map_progress = 0.0

    # Map publish performance caps
    state.max_publish_points = 600000
    state.map_publish_every_n = 3

    # CPU⇄GPU toggle
    state.use_gpu = bool(HAVE_CUPOCH)

    last_map_version = {"v": -1}
    last_published_version = {"v": -1}
    pushing = {"v": False}
    last_fast = {"v": False}

    # Prefetch
    PREFETCH = 24
    pool = ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2))
    def prefetch(idx):
        end = min(idx + 1 + PREFETCH, state.n_frames)
        for j in range(idx + 1, end): pool.submit(load_frame, j)

    # ---- Disable alpha/peeling for splats (robust) ----
    def configure_alpha_features(render_window, renderers, enable: bool):
        try: render_window.SetMultiSamples(0)
        except Exception: pass
        try: render_window.SetAlphaBitPlanes(0)
        except Exception: pass
        for r in renderers:
            try:
                r.SetUseDepthPeeling(0)
                r.SetMaximumNumberOfPeels(0)
                r.SetOcclusionRatio(0.0)
            except Exception:
                pass
        try:
            ren_marker.SetUseDepthPeeling(0)
            ren_marker.SetMaximumNumberOfPeels(0)
            ren_marker.SetOcclusionRatio(0.0)
        except Exception:
            pass

    # SLAM worker
    slam: Optional[SimpleSLAM] = None
    def make_slam() -> SimpleSLAM:
        def pp(idx, min_d, max_d, voxel):
            return preprocess_frame(idx, float(min_d), float(max_d), float(voxel))
        s = SimpleSLAM(
            files, preprocess_fn=pp,
            map_voxel=float(state.map_voxel), icp_voxel=float(state.icp_voxel),
            loop_gap=int(state.loop_gap), loop_thresh=float(state.loop_thresh),
            optimize_every=int(state.optimize_every),
            enable_loop=bool(state.loop_on), rebuild_after_opt=bool(state.rebuild_after_opt),
            loop_stride=int(state.loop_stride),
        )
        s.min_d = float(state.min_d); s.max_d = float(state.max_d)
        s.ds_voxel = float(state.ds_voxel); s.loop_on = bool(state.loop_on)
        return s

    def push_view():
        if not pushing["v"]:
            pushing["v"] = True
            try: ctrl.view_update()
            finally: pushing["v"] = False

    def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        if points.size == 0: return points
        P = points[:, :3]; Pw = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        out = np.zeros_like(points); out[:, :3] = Pw; out[:, 3] = points[:, 3] if points.shape[1] > 3 else 0.0
        return out

    # ---- GPU splats self-test ----
    def gpu_splats_self_test(poly_for_test=None) -> bool:
        try:
            if poly_for_test is None:
                test = np.array([[0,0,0,1],[1,0,0,1],[0,1,0,1]], np.float32)
                poly_for_test = np_points_to_polydata(test, state.color_mode, (0.0, 1.0))
            rw.Render()
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(rw)
            w2i.ReadFrontBufferOff()
            w2i.Update()
            img = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars())
            return bool(img is not None and img.size and img.max() > 0)
        except Exception:
            return False

    # ---- Mapper swapping (robust) ----
    def swap_map_mapper():
        want_fast = bool(state.use_fast_mapper)
        is_fast   = isinstance(map_actor.GetMapper(), vtk.vtkPointGaussianMapper)

        if want_fast and not is_fast:
            if IS_WINDOWS:
                try: rw.SetOffScreenRendering(0)
                except Exception: pass
            configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
            try: rw.SetMultiSamples(0)
            except Exception: pass

            map_actor.SetMapper(map_mapper_fast)
            map_actor.GetProperty().SetOpacity(1.0)
            map_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))

            ok = gpu_splats_self_test(map_mapper_fast.GetInput())
            if not ok:
                map_actor.SetMapper(map_mapper_std)
                configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
                map_actor.GetProperty().SetOpacity(float(state.map_opacity))
                state.use_fast_mapper = False
                print("[WARN] Fast GPU splats failed; reverting to standard mapper.")

        elif not want_fast and is_fast:
            map_actor.SetMapper(map_mapper_std)
            configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
            map_actor.GetProperty().SetOpacity(float(state.map_opacity))

    def swap_scan_mapper():
        want_fast = bool(state.use_fast_mapper)
        is_fast   = isinstance(scan_actor.GetMapper(), vtk.vtkPointGaussianMapper)

        if want_fast and not is_fast:
            if IS_WINDOWS:
                try: rw.SetOffScreenRendering(0)
                except Exception: pass
            configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
            try: rw.SetMultiSamples(0)
            except Exception: pass

            scan_actor.SetMapper(scan_mapper_fast)
            scan_actor.GetProperty().SetOpacity(1.0)
            scan_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))

            ok = gpu_splats_self_test(scan_mapper_fast.GetInput())
            if not ok:
                scan_actor.SetMapper(scan_mapper_std)
                configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
                scan_actor.GetProperty().SetOpacity(1.0)
                state.use_fast_mapper = False
                print("[WARN] Fast GPU splats failed; reverting to standard mapper.")

        elif not want_fast and is_fast:
            scan_actor.SetMapper(scan_mapper_std)
            configure_alpha_features(rw, (ren_left, ren_right, ren_marker), False)
            scan_actor.GetProperty().SetOpacity(1.0)

    def iw_for_render():
        if state.auto_intensity:
            return (1.0, 0.0)  # hi<=lo => auto
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

    # ---- Pose marker (persistent, overlay, depth test off) ----
    def update_pose_marker():
        if not bool(state.show_pose_marker):
            marker_actor.SetVisibility(False)
            return

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
                marker_actor.SetVisibility(False)
                return
        else:
            _last_pose["T"] = T

        p, fwd, left, up = extract_pose_vectors(T)
        R = np.column_stack((fwd, left, up))
        Tw = np.eye(4, dtype=np.float32); Tw[:3, :3] = R; Tw[:3, 3] = p
        marker_actor.SetUserMatrix(np_to_vtk_matrix(Tw))

        s = float(as_float(state.marker_size, 1.0))
        if bool(state.marker_auto_scale):
            cam = ren_right.GetActiveCamera()
            cpos = np.array(cam.GetPosition(), dtype=float)
            dist = max(1.0, float(np.linalg.norm(cpos - p)))
            s = max(
                float(as_float(state.marker_size_min, 0.6)),
                min(
                    float(as_float(state.marker_size, 1.0)),
                    float(as_float(state.marker_scale_k, 0.03)) * dist,
                ),
            )
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

    # ---- Update functions ----
    def update_scan_actor(idx: int):
        pts = preprocess_frame(idx, float(state.min_d), float(state.max_d), float(state.ds_voxel))
        if state.use_world_pose and slam is not None:
            with slam.lock:
                if idx < len(slam.poses): pts = transform_points(pts, slam.poses[idx])
        s = stride_for_scan()
        if s > 1 and len(pts) > 0: pts = pts[::s]
        const = (255, 240, 60) if state.highlight_scan else None
        poly = np_points_to_polydata(pts, state.color_mode, iw_for_render(), constant_rgb=const)

        scan_mapper_std.SetInputData(poly)
        scan_mapper_fast.SetInputData(poly)
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
        if slam is None: return
        with slam.lock:
            ver = slam.map_version
            prog = float(slam.progress)
            running = bool(slam.running)
            pts = slam.latest_map
        state.map_progress = prog; state.map_building = running

        # Publish-skipping while building
        if running and last_published_version["v"] >= 0:
            if (ver - last_published_version["v"]) < max(1, as_int(state.map_publish_every_n, 3)):
                return

        if ver == last_map_version["v"]:
            return
        last_map_version["v"] = ver

        # Cap publish size with adaptive stride
        s = stride_for_map()
        cap = max(100_000, as_int(state.max_publish_points, 600000))
        if len(pts) > cap:
            s = max(s, int(math.ceil(len(pts) / cap)))
        if s > 1 and len(pts) > 0:
            pts = pts[::s]

        poly = np_points_to_polydata(pts, state.color_mode, iw_for_render())
        map_mapper_std.SetInputData(poly)
        map_mapper_fast.SetInputData(poly)
        if isinstance(map_actor.GetMapper(), vtk.vtkPointGaussianMapper):
            map_mapper_fast.SetScaleFactor(max(0.05, as_float(state.splat_size)))
        else:
            div = 2 if state.interacting else 1
            ps_map = max(1, as_int(state.point_size_map))
            map_actor.GetProperty().SetPointSize(max(1, ps_map // div))
            map_actor.GetProperty().SetOpacity(float(state.map_opacity))
        swap_map_mapper()

        if len(pts) > 0:
            fit_camera_once(ren_right, side="right", margin=2.4)

        last_published_version["v"] = ver

    def update_frame():
        idx = int(state.frame)
        if state.show_current: update_scan_actor(idx)
        update_map_actor()
        update_camera_follow_right()
        update_view_layout()
        update_pose_marker()

        # Validate fast splats on first use
        if state.use_fast_mapper and not last_fast["v"]:
            if IS_WINDOWS:
                try: rw.SetOffScreenRendering(0)
                except Exception: pass
            ok = gpu_splats_self_test()
            if not ok:
                state.use_fast_mapper = False
                swap_map_mapper(); swap_scan_mapper()
                print("[WARN] Fast GPU splats not supported; reverted to standard mapper.")
            last_fast["v"] = bool(state.use_fast_mapper)
        elif not state.use_fast_mapper and last_fast["v"]:
            last_fast["v"] = False

        push_view(); prefetch(idx)

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
        update_frame()

    # Fit both views
    @ctrl.trigger("fit_view")
    def _fit_view():
        fit_camera(ren_left, margin=2.2)
        fit_camera(ren_right, margin=2.4)
        ctrl.view_update()

    # interaction flag (from JS)
    @ctrl.trigger("set_interacting")
    def _set_interacting(flag=0):
        prev = bool(state.interacting)
        state.interacting = bool(int(flag))
        if prev != state.interacting:
            update_frame()

    # CPU⇄GPU toggle handler
    @ctrl.trigger("toggle_gpu")
    def _toggle_gpu():
        global USE_GPU
        state.use_gpu = bool(state.use_gpu and HAVE_CUPOCH)
        USE_GPU = bool(state.use_gpu)
        try: preprocess_frame.cache_clear()
        except Exception: pass
        # Clean switch for SLAM
        nonlocal slam
        try:
            if slam is not None and slam.is_alive():
                slam.stop(); time.sleep(0.05)
                if state.slam_on:
                    slam = make_slam(); slam.start()
        except Exception:
            pass
        print(f"[INFO] ICP backend: {'GPU (Cupoch)' if (USE_GPU and HAVE_CUPOCH) else 'CPU (Open3D)'}")

    # SLAM control
    def start_stop_slam():
        nonlocal slam
        if state.slam_on:
            if slam is None or not slam.is_alive():
                slam = make_slam(); slam.start()
        else:
            if slam is not None and slam.is_alive():
                slam.stop()

    ctrl.start_stop_slam = start_stop_slam

    def build_full_map():
        """Start SLAM from scratch; cap publish size; relax optimization cadence."""
        nonlocal slam, last_published_version
        empty = np_points_to_polydata(np.empty((0, 4), np.float32), state.color_mode, (0.0, 1.0))
        map_mapper_std.SetInputData(empty); map_mapper_fast.SetInputData(empty)
        ctrl.view_update()

        if slam is not None and slam.is_alive():
            slam.stop(); time.sleep(0.05)
        n = len(files)
        state.optimize_every = max(200, n // 3)
        state.loop_stride    = max(2, state.loop_stride)
        state.map_voxel      = max(0.35, as_float(state.map_voxel, 0.3))

        slam = make_slam()
        state.map_building = True; state.map_progress = 0.0
        last_map_version["v"] = -1
        last_published_version["v"] = -1
        _first_fit_done["left"] = False; _first_fit_done["right"] = False
        slam.start()
        update_view_layout(); ctrl.view_update()

    ctrl.build_full_map = build_full_map

    def force_rebuild():
        nonlocal slam
        if slam is None: return
        slam.rebuild_map()
        _refresh_map()

    ctrl.force_rebuild = force_rebuild

    def reset_map():
        nonlocal slam
        if slam is not None and slam.is_alive(): slam.stop(); time.sleep(0.05)
        slam = None; last_map_version["v"] = -1; last_published_version["v"] = -1
        _first_fit_done["left"] = False; _first_fit_done["right"] = False
        empty = np_points_to_polydata(np.empty((0,4), np.float32), state.color_mode, (0.0, 1.0))
        map_mapper_std.SetInputData(empty); map_mapper_fast.SetInputData(empty)
        scan_mapper_std.SetInputData(empty); scan_mapper_fast.SetInputData(empty)
        map_actor.SetMapper(map_mapper_std); scan_actor.SetMapper(scan_mapper_std)
        marker_actor.SetVisibility(False)
        state.map_progress = 0.0; state.map_building = False
        update_view_layout()
        ctrl.view_update()

    ctrl.reset_map = reset_map

    def _on_client_exited(client_id=None, **_): state.play = False
    try: server.on_client_exited.add(_on_client_exited)
    except Exception: pass

    # ---------- UI ----------
    with SinglePageLayout(server) as layout:
        layout.title.set_text("KITTI LiDAR SLAM (CPU⇄GPU ICP; Split View; Publish Caps)")
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
                        h.Input(type="checkbox", v_model=("map_only_on_build", True), change=ctrl.update_frame); h.Label("Map-only during Build Full Map")
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
                        h.Label("Marker size (cap)"); h.Input(type="range", min=0.3, max=15.0, step=0.1, v_model=("marker_size", 1.0), change=ctrl.update_frame, style="flex:1")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Label("Auto min"); h.Input(type="number", step="0.1", v_model=("marker_size_min", 0.6), change=ctrl.update_frame, style="width:30%")
                        h.Label("Auto scale k"); h.Input(type="number", step="0.005", v_model=("marker_scale_k", 0.03), change=ctrl.update_frame, style="width:40%")

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
                    h.Input(type="range", min=0, max=0.8, step=0.02, v_model=("ds_voxel", 0.2), change=ctrl.update_frame, style="width:100%")

                    h.Hr()
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("use_gpu", HAVE_CUPOCH), change="trigger('toggle_gpu')"); 
                        h.Label(f"Use GPU (Cupoch) for ICP{' (not available)' if not HAVE_CUPOCH else ''}")

                    h.Hr()
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("slam_on", True), change=ctrl.start_stop_slam); h.Label("Run SLAM (live)")
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("loop_on", True)); h.Label("Loop closure enabled")
                    h.Label("Loop gap"); h.Input(type="number", min=10, step=1, v_model=("loop_gap", 30))
                    h.Label("Loop threshold"); h.Input(type="number", step="0.01", v_model=("loop_thresh", 0.25))
                    h.Label("Optimize every N frames"); h.Input(type="number", min=5, step=1, v_model=("optimize_every", 30))
                    h.Label("Loop stride (candidate step)"); h.Input(type="number", min=1, step=1, v_model=("loop_stride", 1))
                    h.Label("Map voxel (m)"); h.Input(type="range", min=0.05, max=1.0, step=0.05, v_model=("map_voxel", 0.3))
                    h.Label("ICP voxel (m)"); h.Input(type="range", min=0.1, max=1.0, step=0.05, v_model=("icp_voxel", 0.4))
                    with h.Div(style="display:flex; gap:8px; align_items:center;"):
                        h.Input(type="checkbox", v_model=("rebuild_after_opt", True)); h.Label("Rebuild map after optimization")

                    h.Hr()
                    h.Label("Map publish caps (for performance)")
                    with h.Div(style="display:flex; gap:8px; align-items:center;"):
                        h.Label("Max points to publish"); h.Input(type="number", min=100000, step=50000, v_model=("max_publish_points", 600000), change=ctrl.update_frame, style="width:40%")
                        h.Label("Publish every N versions"); h.Input(type="number", min=1, step=1, v_model=("map_publish_every_n", 3), change=ctrl.update_frame, style="width:30%")

                    with h.Div(style="display:flex; gap:8px; margin-top:8px;"):
                        h.Button("Build Full Map (run to end)", click=ctrl.build_full_map, style="flex:2")
                        h.Button("Force Rebuild Now", click=ctrl.force_rebuild, style="flex:1")
                        h.Button("Reset Map/SLAM", click=ctrl.reset_map, style="flex:1")

                    with h.Div(style="margin-top:6px; font-size:12px; opacity:0.8;"):
                        h.Div(children=["Map: ", h.Span(children=["{{ (map_progress*100).toFixed(1) }}%"]), "  ", h.Span(children=["{{ map_building ? '(running)' : '' }}"])])
                        # Pause refresh during interaction; throttle server pushes
                        h.Script("""
                        (function(){
                          const root = document.querySelector('.vtk-wrap');
                          if (!root || window._pauseHooksSet) return;
                          window._pauseHooksSet = true;

                          function stopTimer(){ if (window._mapTimer){ clearInterval(window._mapTimer); window._mapTimer = null; } }
                          function startTimer(){ if (!window._mapTimer){ window._mapTimer = setInterval(()=>{ try{ trigger('refresh_map'); }catch(e){} }, 700); } }

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
                        interactive_quality=35,
                        still_ratio=1,
                        still_quality=60,
                        max_image_size=720,
                    ) as view:
                        def _safe_update():
                            try: view.update()
                            except Exception: return
                        ctrl.view_update = _safe_update
                        ctrl.update_frame()
                        ctrl.apply_interactor_style()

    # Start SLAM immediately if requested at startup
    if state.slam_on:
        try: ctrl.start_stop_slam()
        except Exception: pass

    return server

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    server = build_app(args.seq_dir, fps=args.fps)
    server.start(port=args.port)
