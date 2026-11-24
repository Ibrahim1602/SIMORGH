"""Loaders for one CityFlowV2 camera folder, e.g. train/S01/c001.

Everything the dataset ships for a camera is read here:
  gt/gt.txt, det/*.txt, mtsc/*.txt  -> load_mot()
  segm/segm_mask_rcnn.txt           -> Masks
  calibration.txt                   -> Calibration (pixel -> GPS, ground metres)
  roi.jpg                           -> load_roi() / roi_fraction()
plus the README's frame rate and scenario-centre GPS facts.
"""
import ast
import re
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

# README: every video is 10 FPS except c015 in S03, which is 8 FPS.
FPS = {"c015": 8.0}
DEFAULT_FPS = 10.0

# README: approximate GPS centre of each scenario (exact camera positions are not given).
SCENARIO_CENTER = {
    "S01": (42.525678, -90.723601),
    "S02": (42.491916, -90.723723),
    "S03": (42.498780, -90.686393),
    "S04": (42.498780, -90.686393),
    "S05": (42.498780, -90.686393),
    "S06": (42.492448, -90.723343),
}


# ----------------------------------------------------------------------------
# MOTChallenge text files: gt, det, mtsc
# ----------------------------------------------------------------------------
def load_mot(path):
    """MOTChallenge text -> {frame: array of rows [id, x, y, w, h, conf]}.

    x, y is the top-left corner in pixels; frames start at 1.
    gt: id = vehicle identity (shared across cameras), conf = 1.
    det: id = -1 (not tracked yet).  mtsc: id = track id local to this camera.
    """
    a = np.loadtxt(path, delimiter=",", usecols=range(7), ndmin=2)
    a = a[np.argsort(a[:, 0], kind="stable")]
    frames, starts = np.unique(a[:, 0], return_index=True)
    return {int(f): chunk[:, 1:] for f, chunk in zip(frames, np.split(a, starts[1:]))}


def iou_matrix(a, b):
    """a: (n, 4), b: (m, 4) as x, y, w, h  ->  (n, m) intersection-over-union."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


# ----------------------------------------------------------------------------
# Region of interest
# ----------------------------------------------------------------------------
def load_roi(path):
    """roi.jpg -> boolean array, True where the mask is white."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def roi_fraction(roi, x, y, w, h):
    """Share of the box (x, y, w, h) that lies inside the ROI."""
    x1, y1 = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x + w), roi.shape[1]), min(int(y + h), roi.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(roi[y1:y2, x1:x2].mean())


# ----------------------------------------------------------------------------
# Calibration: pixel -> GPS -> metres
# ----------------------------------------------------------------------------
def _numbers(text):
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text)])


class Calibration:
    """calibration.txt: line 1 is a 3x3 homography; optional intrinsics and
    distortion coefficients; last a reprojection error in pixels.

    The homography stored in the file maps GPS -> pixel, so pixel -> GPS uses its
    inverse (checked on c001: the image centre lands ~12 m from the README's S01 centre).
    """

    def __init__(self, path, image_size=(1920, 1080)):
        lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        rows = lines[0].split(":", 1)[1].split(";")
        self.H = np.array([_numbers(r) for r in rows])
        self.Hinv = np.linalg.inv(self.H)
        self.K = self.dist = self.error = None
        for line in lines[1:]:
            if ":" not in line:
                continue
            low, nums = line.lower(), _numbers(line.split(":", 1)[1])
            try:
                if "reprojection" in low:
                    self.error = float(nums[0])
                elif "intrinsic" in low:
                    self.K = nums.reshape(3, 3)
                elif "distortion" in low:
                    self.dist = nums
            except (ValueError, IndexError):
                pass          # unfamiliar layout: keep going with the homography only
        w, h = image_size
        self.size = image_size
        self.lat0, self.lon0 = self.pixel_to_gps(w / 2, h / 2)
        self.m_per_deg_lat = 111_132.0
        self.m_per_deg_lon = 111_320.0 * np.cos(np.radians(self.lat0))

    def pixel_to_gps(self, x, y):
        if self.K is not None and self.dist is not None:       # undo radial distortion first
            pt = np.array([[[x, y]]], dtype=np.float64)
            x, y = cv2.undistortPoints(pt, self.K, self.dist, P=self.K)[0, 0]
        p = self.Hinv @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])           # (lat, lon)

    def pixels_to_gps(self, pts):
        """Many pixels at once: (n, 2) -> lat, lon (arrays of n) and `ok`, False for pixels above the horizon.

        Those point at no ground at all, and the maths turns them into places behind the camera, so they must
        not be used. The bottom of the image always shows ground; `ok` means "on the same side of the horizon".
        """
        w, h = self.size
        pts = np.vstack([np.asarray(pts, dtype=np.float64).reshape(-1, 2), [[w / 2, h - 1]]])
        if self.K is not None and self.dist is not None:
            pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        p = np.c_[pts, np.ones(len(pts))] @ self.Hinv.T
        ok = np.sign(p[:-1, 2]) == np.sign(p[-1, 2])
        return p[:-1, 0] / p[:-1, 2], p[:-1, 1] / p[:-1, 2], ok

    def ground_xy(self, x, y):
        """Pixel -> metres east/north of the projected image centre."""
        lat, lon = self.pixel_to_gps(x, y)
        return (lon - self.lon0) * self.m_per_deg_lon, (lat - self.lat0) * self.m_per_deg_lat

    def distance_to_scenario_centre(self, scenario):
        """Sanity check: metres from the projected image centre to the README's scenario centre."""
        if scenario not in SCENARIO_CENTER:
            return None
        lat, lon = SCENARIO_CENTER[scenario]
        return float(np.hypot((lon - self.lon0) * self.m_per_deg_lon, (lat - self.lat0) * self.m_per_deg_lat))


class SpeedEstimator:

    def __init__(self, calib, fps, window=10, min_span=4):
        self.calib, self.fps, self.min_span = calib, fps, min_span
        self.hist = defaultdict(lambda: deque(maxlen=window + 1))   # tid -> (frame, gx, gy)
        self.speeds = defaultdict(list)

    def update(self, tid, frame, x, y):
        """x, y = the point where the vehicle touches the road (bottom-centre of its box)."""
        gx, gy = self.calib.ground_xy(x, y)
        h = self.hist[tid]
        h.append((frame, gx, gy))
        f0, x0, y0 = h[0]
        if frame - f0 < self.min_span:
            return None
        kmh = float(np.hypot(gx - x0, gy - y0) / ((frame - f0) / self.fps) * 3.6)
        self.speeds[tid].append(kmh)
        return kmh


# ----------------------------------------------------------------------------
# Mask R-CNN masks (segm_mask_rcnn.txt)
# ----------------------------------------------------------------------------
def rle_decode(counts, h, w):
    """Decode a COCO compressed run-length string into an (h, w) boolean mask."""
    runs, p = [], 0
    while p < len(counts):
        x, k, more = 0, 0, True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    flat = np.zeros(h * w, dtype=bool)
    pos, on = 0, False
    for r in runs:
        if on:
            flat[pos:pos + r] = True
        pos += r
        on = not on
    return flat.reshape((h, w), order="F")        # COCO stores masks column by column


class Masks:
    """One mask per Mask R-CNN detection, looked up by overlap with any box."""

    def __init__(self, path, min_conf=0.5):
        self.by_frame = defaultdict(list)          # frame -> [(box, counts, h, w)]
        with open(path, encoding="ascii", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\r\n").split(",", 10)
                conf = float(parts[6])
                if conf < min_conf:
                    continue
                d = parts[10]                       # {'counts': '...', 'size': [1080L, 1920L]}
                start, end = d.index("'counts': ") + len("'counts': "), d.rindex(", 'size'")
                counts = ast.literal_eval(d[start:end])
                h, w = (int(v) for v in re.findall(r"\d+", d[end:]))
                box = tuple(float(v) for v in parts[2:6])
                self.by_frame[int(parts[0])].append((box, counts, h, w))

    def mask_for(self, frame, box, min_iou=0.5):
        """Full-frame boolean mask of the detection that best overlaps `box` (x, y, w, h), or None."""
        items = self.by_frame.get(frame)
        if not items:
            return None
        ious = iou_matrix(np.array([box], dtype=float), np.array([it[0] for it in items]))[0]
        i = int(ious.argmax())
        if ious[i] < min_iou:
            return None
        _, counts, h, w = items[i]
        return rle_decode(counts, h, w)


# ----------------------------------------------------------------------------
# Vehicle crops with the background removed (Mask R-CNN masks)
# ----------------------------------------------------------------------------
MASK_FILL = (104, 116, 122)       # BGR of CLIP's mean colour: after CLIP's own normalisation the background becomes "nothing"


def masked_crop(frame, box, masks, frame_no, min_fill=0.1):
    """The vehicle inside box (x, y, w, h) with everything outside its Mask R-CNN mask painted grey, or None when no mask matches
    the box (or the matching mask covers under min_fill of it, which means it belongs to something else)."""
    H, W = frame.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[0] + box[2]), W), min(int(box[1] + box[3]), H)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    full = masks.mask_for(frame_no, box)
    if full is None:
        return None
    m = full[y1:y2, x1:x2]
    if m.mean() < min_fill:
        return None
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)      # keep a thin rim: masks are slightly tight
    out = frame[y1:y2, x1:x2].copy()
    out[~m] = MASK_FILL
    return out


# ----------------------------------------------------------------------------
# Spatio-temporal gating: can one vehicle be this track in one camera and that track in another?
# ----------------------------------------------------------------------------
# A track is reduced to when and where it enters and leaves its camera and which way it moves, in metres east/north of one
# origin shared by the whole scenario. Two tracks of different cameras are then compared by physics only: a vehicle cannot be
# in two places at once, cannot drive faster than a vehicle can, and leaves one camera heading towards the other, not away.
# These are defaults; finetune.py fits them to the labelled scenarios and writes gate.json.
GATE = dict(slack=2.0,        # seconds of clock error tolerated (frames are dropped in the videos, the clocks are not exact)
            overlap_m=15.0,   # cameras that see the same moment: the positions may differ this much (calibration error); 2x = impossible
            max_gap=120.0,    # seconds between leaving one camera and entering the next; 2x = impossible
            max_speed=30.0,   # m/s needed to cover the distance between the two cameras in that time; 2x = impossible
            min_dist=25.0,    # the direction test only applies when the cameras are at least this far apart (metres)
            min_speed=1.0,    # ...and when the vehicle is actually moving (m/s)
            cos_ok=0.0,       # heading versus direction to the other camera: no penalty at or above this cosine
            cos_bad=-0.6)     # ...impossible at or below this cosine (moving away)
_VEC = ("pm", "vm", "p0", "p1", "vin", "vout")
_NONE = {"ok": False, "t0": 0.0, "t1": 0.0, "tm": 0.0, **{k: (0.0, 0.0) for k in _VEC}}


def to_metres(lat, lon, origin):
    """(lat, lon) -> metres east and north of origin = (lat0, lon0). Use the same origin for every camera of a scenario."""
    lat0, lon0 = origin
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 111_132.0


def summarize_track(t, xy, win=2.0, min_points=4, min_duration=0.5):
    """t: seconds (increasing), xy: (n, 2) ground positions in metres -> the summary used by the gate, or None for a track too short
    to say anything (the gate then has no opinion about it). Positions and velocities come from straight-line fits, which smooths
    the jitter of box bottoms."""
    t, xy = np.asarray(t, float), np.asarray(xy, float).reshape(-1, 2)
    if len(t) < min_points or t[-1] - t[0] < min_duration:
        return None

    def line(mask):                       # least-squares line through the points: mean time, mean position, velocity (or None)
        tt, pp = t[mask], xy[mask]
        tm, pm = tt.mean(), pp.mean(axis=0)
        var = ((tt - tm) ** 2).sum()
        v = ((tt - tm)[:, None] * (pp - pm)).sum(axis=0) / var if len(tt) >= 3 and var > 1e-6 else None
        return tm, pm, v

    tm, pm, vm = line(np.ones(len(t), bool))
    vm = np.zeros(2) if vm is None else vm
    out = {"ok": True, "t0": t[0], "t1": t[-1], "tm": tm, "pm": pm, "vm": vm}
    for pos, vel, mask, at in (("p0", "vin", t <= t[0] + win, t[0]), ("p1", "vout", t >= t[-1] - win, t[-1])):
        tw, pw, vw = line(mask)
        vw = vm if vw is None else vw
        out[pos], out[vel] = pw + vw * (at - tw), vw          # position exactly at the entry / exit time
    return out


def stack_tracks(tracks):
    """List of summaries (None allowed) -> dict of arrays, one row per track; `ok` is False for the None ones."""
    rows = [r if r is not None else _NONE for r in tracks]
    out = {}
    for k in _NONE:
        if k == "ok":
            out[k] = np.array([r[k] for r in rows], bool)
        elif k in _VEC:
            out[k] = np.array([r[k] for r in rows], float).reshape(len(rows), 2)
        else:
            out[k] = np.array([r[k] for r in rows], float)
    return out


def pair_quantities(A, B, outer=True, p=None):
    """The measurements the gate looks at. outer=True: every track of A against every track of B -> arrays (n, m).
    outer=False: A[i] against B[i] (equal lengths) -> arrays (n,)."""
    p = {**GATE, **(p or {})}
    a = lambda k: (A[k][:, None] if outer else A[k])
    b = lambda k: (B[k][None] if outer else B[k])
    vec = lambda x: x[..., None]
    ok = a("ok") & b("ok")
    lo, hi = np.maximum(a("t0"), b("t0")), np.minimum(a("t1"), b("t1"))
    overlap = (hi - lo) > p["slack"]
    # same moment in both cameras: how far apart are the two straight-line models at the start, middle and end of it?
    d_ov = np.median([np.linalg.norm((a("pm") + a("vm") * vec(tt - a("tm"))) - (b("pm") + b("vm") * vec(tt - b("tm"))), axis=-1)
                      for tt in (lo, (lo + hi) / 2, hi)], axis=0)
    # one after the other: leave the first camera, enter the second
    a_first = a("tm") < b("tm")
    pick = lambda x, y: np.where(vec(a_first), x, y)
    gap = np.where(a_first, b("t0") - a("t1"), a("t0") - b("t1"))
    p_exit, p_entry = pick(a("p1"), b("p1")), pick(b("p0"), a("p0"))
    v_exit, v_entry = pick(a("vout"), b("vout")), pick(b("vin"), a("vin"))
    d = np.linalg.norm(p_entry - p_exit, axis=-1)
    to_second = (p_entry - p_exit) / vec(np.maximum(d, 1e-6))
    sp_exit, sp_entry = np.linalg.norm(v_exit, axis=-1), np.linalg.norm(v_entry, axis=-1)
    return {"ok": ok, "overlap": overlap, "d_ov": d_ov, "gap": gap, "d": d,
            "v_need": d / np.maximum(gap, p["slack"]),
            "cos_exit": (v_exit * to_second).sum(-1) / np.maximum(sp_exit, 1e-6), "sp_exit": sp_exit,
            "cos_entry": (v_entry * to_second).sum(-1) / np.maximum(sp_entry, 1e-6), "sp_entry": sp_entry}


def gate_penalty(q, p=None):
    """From pair_quantities to a penalty per pair: 0 = plausible, 1 = impossible, in between = unlikely. Pairs with a track the gate
    knows nothing about (too short) get 0."""
    p = {**GATE, **(p or {})}
    ramp = lambda x: np.clip(x, 0.0, 1.0)
    far = ramp((q["gap"] - p["max_gap"]) / p["max_gap"])
    fast = ramp((q["v_need"] - p["max_speed"]) / p["max_speed"])

    def away(cos, speed):                 # moving away from the other camera (only judged when it is far enough and moving)
        judged = (q["d"] > p["min_dist"]) & (speed > p["min_speed"])
        return np.where(judged, ramp((p["cos_ok"] - cos) / (p["cos_ok"] - p["cos_bad"])), 0.0)

    one_after_other = np.maximum.reduce([far, fast, away(q["cos_exit"], q["sp_exit"]), away(q["cos_entry"], q["sp_entry"])])
    same_moment = ramp((q["d_ov"] - p["overlap_m"]) / p["overlap_m"])
    return np.where(q["ok"], np.where(q["overlap"], same_moment, one_after_other), 0.0)
