"""
detection_tracking.py
Modular detection + tracking + trajectory feature collection.

Provides:
    - DetectorTracker: wraps YOLOv8 + (DeepSORT or ByteTrackLite fallback)
    - ByteTrackLite: minimal IOU tracker
    - helper functions for geometric / entropy statistics

Usage inside main fusion:
    from detection_tracking import DetectorTracker

NOTE:
    pip install ultralytics deep-sort-realtime opencv-python numpy torch torchvision pandas
"""

from __future__ import annotations
import math, numpy as np, cv2, torch
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# ----------------- Optional imports -----------------
try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Install ultralytics: pip install ultralytics") from e

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except ImportError:
    HAS_DEEPSORT = False


# ----------------- Lightweight ByteTrack-like tracker -----------------
class ByteTrackLite:
    def __init__(self, iou_thresh=0.45, keep_alive=30):
        self.iou_thresh = iou_thresh
        self.keep_alive = keep_alive
        self.next_id = 1
        self.tracks = {}  # id -> dict(box, score, age, last_seen)

    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0, x2 - x1); ih = max(0, y2 - y1)
        inter = iw * ih
        if inter == 0: return 0.0
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, det_boxes, det_scores):
        updated = set()
        for db, sc in zip(det_boxes, det_scores):
            best_iou, best_id = 0.0, None
            for tid, info in self.tracks.items():
                i = self.iou(db, info["box"])
                if i > best_iou:
                    best_iou, best_id = i, tid
            if best_iou >= self.iou_thresh:
                t = self.tracks[best_id]
                t["box"] = db
                t["score"] = sc
                t["last_seen"] = 0
                updated.add(best_id)
            else:
                self.tracks[self.next_id] = {"box": db, "score": sc, "age": 0, "last_seen": 0}
                updated.add(self.next_id)
                self.next_id += 1
        # aging
        to_del = []
        for tid, info in self.tracks.items():
            if tid not in updated:
                info["last_seen"] += 1
            info["age"] += 1
            if info["last_seen"] > self.keep_alive:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        active = []
        for tid, info in self.tracks.items():
            x1,y1,x2,y2 = info["box"]
            active.append({"track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": info["score"]})
        return active


# ----------------- Geometry & stats helpers -----------------
def direction_entropy(vectors, bins=12):
    vecs = [(dx,dy) for dx,dy in vectors if dx or dy]
    if not vecs: return 0.0
    angles = [math.atan2(dy,dx) for dx,dy in vecs]
    hist,_ = np.histogram(angles, bins=bins, range=(-math.pi,math.pi))
    p = hist.astype(float)
    if p.sum() == 0: return 0.0
    p /= p.sum()
    ent = -(p[p>0]*np.log(p[p>0])).sum()
    return ent / math.log(bins)

def mean_pairwise_distance(centroids, sample_max=60):
    n = len(centroids)
    if n <= 1: return 0.0
    idx = np.arange(n)
    if n > sample_max:
        idx = np.random.choice(idx, sample_max, replace=False)
    pts = np.array([centroids[i] for i in idx])
    if pts.shape[0] < 2: return 0.0
    diff = pts[:,None,:] - pts[None,:,:]
    d = np.sqrt((diff**2).sum(-1))
    return float(d[np.triu_indices(len(pts),1)].mean())

def velocity_histogram(speeds, bins=8):
    if not speeds:
        return np.zeros(bins)
    s = np.array(speeds)
    hist,_ = np.histogram(s, bins=bins, range=(0, max(1.0, s.max())))
    if hist.sum() == 0: return np.zeros(bins)
    return hist / hist.sum()

def density_grid(centroids, W, H, g=4):
    grid = np.zeros((g,g), dtype=float)
    if not centroids: return grid
    for x,y in centroids:
        gx = min(g-1, max(0, int((x/W)*g)))
        gy = min(g-1, max(0, int((y/H)*g)))
        grid[gy,gx] += 1
    grid /= (grid.sum() + 1e-6)
    return grid


# ----------------- NMS merge for multi-scale fusion -----------------
def nms_merge(boxes, scores, iou_thresh=0.55):
    if not boxes: return [],[]
    b = np.array(boxes); s = np.array(scores)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        ious = batch_iou(b[i], b[order[1:]])
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds+1]
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def batch_iou(one, many):
    x1 = np.maximum(one[0], many[:,0]); y1 = np.maximum(one[1], many[:,1])
    x2 = np.minimum(one[2], many[:,2]); y2 = np.minimum(one[3], many[:,3])
    iw = np.clip(x2-x1,0,None); ih = np.clip(y2-y1,0,None)
    inter = iw*ih
    area1 = (one[2]-one[0])*(one[3]-one[1])
    area2 = (many[:,2]-many[:,0])*(many[:,3]-many[:,1])
    return inter / (area1 + area2 - inter + 1e-6)


# ----------------- Detector + Tracker wrapper -----------------
class DetectorTracker:
    """
    Handles YOLOv8 detection + chosen tracker (DeepSORT or ByteTrackLite).
    Provides per-frame tracking output for fusion module.

    get_tracks(frame_idx, frame_bgr) -> list of dicts:
        {track_id, x1,y1,x2,y2,cx,cy,score}

    Multi-scale: pass dual_scale=True; will run second scale and NMS merge.

    """
    def __init__(self,
                 model_path="yolov8n.pt",
                 device="cpu",
                 conf=0.35,
                 classes=(0,),
                 imgsz_base=640,
                 dual_scale=False,
                 big_scale=896,
                 tracker_type="bytetrack",
                 max_age=30,
                 n_init=3):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.classes = tuple(classes)
        self.imgsz_base = imgsz_base
        self.dual_scale = dual_scale
        self.big_scale = big_scale
        self.tracker_type = tracker_type

        if tracker_type == "deepsort":
            if not HAS_DEEPSORT:
                raise RuntimeError("DeepSORT not installed. Install or use tracker_type='bytetrack'")
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.4,
                embedder="mobilenet",
                half=(device not in ["cpu","mps"]),
                bgr=True,
            )
            self.use_deepsort = True
        else:
            self.tracker = ByteTrackLite(iou_thresh=0.45, keep_alive=max_age)
            self.use_deepsort = False

        # state for speed
        self.last_pos = {}
        self.track_vectors = defaultdict(list)
        self.track_speeds = defaultdict(list)
        self.track_accels = defaultdict(list)
        self.last_speed = {}

    def _detect(self, frame):
        boxes_all, scores_all = [], []
        scales = [self.imgsz_base]
        if self.dual_scale and self.big_scale != self.imgsz_base:
            scales.append(self.big_scale)

        for sz in scales:
            res = self.model.predict(
                source=frame,
                imgsz=sz,
                verbose=False,
                conf=self.conf,
                device=self.device
            )[0]
            for b in res.boxes:
                cls = int(b.cls.item())
                if self.classes and cls not in self.classes: continue
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf.item())
                boxes_all.append(xyxy)
                scores_all.append(conf)

        if len(boxes_all) > 0:
            boxes_all, scores_all = nms_merge(boxes_all, scores_all)
        return boxes_all, scores_all

    def get_tracks(self, frame_idx: int, frame_bgr) -> List[Dict]:
        boxes, scores = self._detect(frame_bgr)

        active = []
        if self.use_deepsort:
            ds_in = []
            for (x1,y1,x2,y2), sc in zip(boxes, scores):
                ds_in.append(([x1,y1,x2-x1,y2-y1], sc, "0"))
            tracks = self.tracker.update_tracks(ds_in, frame=frame_bgr)
            for tr in tracks:
                if not tr.is_confirmed(): continue
                x1,y1,x2,y2 = tr.to_ltrb()
                active.append({
                    "track_id": tr.track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": getattr(tr, "det_conf", 1.0)
                })
        else:
            bt = self.tracker.update(boxes, scores)
            active.extend(bt)

        # kinematics
        out = []
        for tr in active:
            x1,y1,x2,y2 = tr["x1"],tr["y1"],tr["x2"],tr["y2"]
            cx = (x1+x2)/2; cy = (y1+y2)/2
            tid = tr["track_id"]
            if tid in self.last_pos:
                px,py = self.last_pos[tid]
                dx,dy = cx-px, cy-py
                dist = math.hypot(dx,dy)
                self.track_vectors[tid].append((dx,dy))
                self.track_speeds[tid].append(dist)
                if tid in self.last_speed:
                    acc = dist - self.last_speed[tid]
                    self.track_accels[tid].append(acc)
                self.last_speed[tid] = dist
            self.last_pos[tid] = (cx,cy)
            out.append({
                "track_id": tid, "x1": x1,"y1": y1,"x2": x2,"y2": y2,
                "cx": cx,"cy": cy,"score": tr["score"]
            })
        return out