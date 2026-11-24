"""Play the cameras of one CityFlowV2 scenario side by side, in sync and in real time, and follow one vehicle.

All cameras share one clock (scenario time, from cam_timestamp), so a vehicle that leaves one camera
shows up in the next at the right moment. Each tile shows the scenario time of the frame it is displaying.

Boxes come from BOXES (see the settings). With BOXES = "tracks:..." the first run detects and tracks every
camera of the scenario that has no track files yet (YOLO, one camera after another) and saves them for next time.

  Click a box        follow that vehicle: it gets a red TARGET box and every other box disappears,
                     and its path appears on the map
  c                  stop following                    Space or k   play / pause
  a or Left          back 5 s                          d or Right   forward 5 s
  r                  restart                           b            boxes on / off
  Esc or q           quit                              Click or drag the bar at the bottom to jump anywhere.

The map under the tiles (north is up) is a satellite photo (aerial photography of the USA from USGS, downloaded once and
kept in TILES_DIR) and shows what each camera sees: the road of its road mask (roi.jpg), placed with its
calibration.txt, as far as one pixel still covers FOOTPRINT_RES metres. Nothing else is plotted until you follow a
vehicle: then its path grows in red as the video plays, a dot marks where it was last seen, and a thin line joins the
pieces seen by different cameras.

Following in the other cameras: with "gt" the dataset knows the same vehicle in every camera. With "tracks:..."
each track gets an appearance vector (CLIP, from its biggest crops) and the most similar track in every other
camera is shown as TARGET with its similarity. With "mtsc:..." only the camera you clicked in can follow.

Spatio-temporal gate (ST_GATE): a match in another camera is only shown if physics allows it: the vehicle cannot be in two places
at once, cannot cover the distance between the cameras faster than a vehicle can drive, and must be leaving one camera towards the
other, not away from it. Positions come from calibration.txt, times from cam_timestamp. The console says what was found where.

Needs for "tracks:...":  py -m pip install ultralytics      (if tracking complains about "lap":  py -m pip install lapx)
Run:  py multicam.py
"""
import json
import math
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import cv2
import numpy as np

from cityflow import (DEFAULT_FPS, FPS, GATE, SCENARIO_CENTER, Calibration, Masks, gate_penalty, load_mot, load_roi, masked_crop,
                      pair_quantities, stack_tracks, summarize_track, to_metres)

# ---- settings ---------------------------------------------------------------
ROOT = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\AICity22_Track1_MTMC_Tracking")
OTHERS = ROOT        # the folder from others.zip: the one that contains cam_timestamp/
TRACKS = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\tracks")   # where the track files of "tracks:..." are kept
TILES_DIR = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\map_tiles")   # satellite tiles are kept here after the first download
SPLIT, SCENARIO = "train", "S01"  # train: S01 S03 S04 | validation: S02 S05 | test: S06
CAMERAS = None                    # None = the scenario's cameras in order, or a list like ["c001", "c003"]
MAX_TILES = 6                     # never show more cameras than this at once
WINDOW = (1280, 760)              # width, height of the window in pixels: video tiles, then the map, then the bar
MAP_H = 240                       # height of the map panel under the tiles
BAR_H = 40
BOXES = "tracks:cityflow_yolo_bytetrack"                      # "gt": human labels (train and validation only, ids shared by all cameras)
                                  # "mtsc:tc_mask_rcnn": the dataset's tracker results (any scenario; ids belong to one camera)
                                  # "tracks:yolo26n_bytetrack": your own: <YOLO model>_<tracker>, made automatically
SHOW_BOXES = True                 # b toggles it while playing
JUMP = 5.0                        # seconds skipped by a / d
MAX_SKIP = 30                     # a camera that is further behind than this many frames seeks instead of catching up
TRACK_IMGSZ = 1280               # detection size for tracking; 640 is about 4x faster but may miss small, far cars
EMBED_CROPS = 5                   # the biggest crops of each track that make up its appearance vector
SATELLITE = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                                  # where the satellite photos come from (any tile address with {z} {x} {y}); None = plain dark map.
                                  # USGS: free, no account, USA only, sharpest level 16 (about 1.8 m per pixel here). Esri's World
                                  # Imagery is sharper (level 19) but has its own terms and may need a key: read them first.
SATELLITE_MAX_ZOOM = 16           # the sharpest level that source has (USGS 16, Esri 19)
SATELLITE_MAX_TILES = 80          # never download more tiles than this for one map
SATELLITE_CREDIT = "USDA, USGS The National Map: Orthoimagery"      # shown on the map; use the credit your source asks for
FOOTPRINT_RES = 0.25              # a camera's footprint on the map ends where one pixel covers more than this many metres
                                  # (0.25 m: a 4.5 m car is about 18 pixels long, the least a detector can follow)
REID_MIN = 0.43                    # matches in other cameras below this similarity are not shown; raise it once you have
                                  # seen what the similarities of right and wrong matches look like
FINETUNE_DIR = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\finetune")   # where finetune.py keeps cityflow_yolo.pt and the re-ID head
HEAD_FILE = "reid_head.npz"       # re-ID head in FINETUNE_DIR ("reid_head_all.npz" = trained on all 5 labelled scenarios); None = plain CLIP
CAMERA_CENTER = True             # subtract each camera's mean appearance vector first. `py finetune.py reid` prints the value to use
RECIPROCAL = True                # show a match only if, looking back from it, the followed vehicle is the best match too (same print-out)
USE_MASKS = True                  # paint the background of every crop grey (Mask R-CNN masks in segm/) before describing it. Must match finetune.py
ST_GATE = True                    # spatio-temporal gate: matches that physics rules out are never shown (needs calibration.txt and cam_timestamp)
ST_WEIGHT = 0.5                   # ...and unlikely ones lose this much similarity per unit of penalty. `py finetune.py reid` prints the value
GATE_FILE = "gate.json"           # thresholds fitted by finetune.py, in FINETUNE_DIR ("gate_all.json" after reid-all); missing = built-in defaults
# -----------------------------------------------------------------------------

WIN = "multicam"
EMPTY = np.zeros((0, 6))
BOX_COLOR = (0, 200, 0)
TARGET_COLOR = (0, 0, 255)        # red (BGR)
DARK_RED = (40, 40, 170)
MAP_COLORS = [(255, 180, 60), (90, 200, 90), (60, 210, 240), (220, 130, 200), (210, 210, 70), (200, 120, 60)]   # no red
BAR_COLOR = (255, 180, 60)
VEHICLES = ("car", "motorcycle", "bus", "truck")
CLIP_MODEL = "openai/clip-vit-base-patch32"
LEFT, RIGHT = {2424832, 65361}, {2555904, 65363}      # arrow keys from cv2.waitKeyEx on Windows and Linux


def boxes_path(folder):
    """Where this camera's box file is, for the BOXES setting."""
    kind, _, name = BOXES.partition(":")
    if kind == "gt":
        return folder / "gt" / "gt.txt"
    if kind == "mtsc":
        return folder / "mtsc" / f"mtsc_{name}.txt"
    if kind == "tracks":
        return TRACKS / name / folder.parent.name / f"{folder.name}.txt"
    raise SystemExit(f'BOXES = "{BOXES}" is not one of "gt", "mtsc:<name>" or "tracks:<name>"')


def fmt(t):
    """Seconds as mm:ss.s"""
    tenths = round(t * 10)
    return f"{tenths // 600:02d}:{(tenths % 600) / 10:04.1f}"


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)   # dark outline, no filled box
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---- tracking: detect, track and describe the vehicles of every camera ---------------------------------------
_clip = []


def embed(crops_bgr):
    """One unit-length CLIP vector per image (rows). This is the only place that knows which model describes a vehicle."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    if not _clip:
        _clip.append((CLIPModel.from_pretrained(CLIP_MODEL).eval(), CLIPProcessor.from_pretrained(CLIP_MODEL)))
    model, proc = _clip[0]
    out = []
    for i in range(0, len(crops_bgr), 64):
        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr[i:i + 64]]
        pixels = proc(images=pil, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            out.append(model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output).numpy())
    f = np.concatenate(out)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


_warned = set()


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


def reid_transform(emb):
    """The stored appearance vectors of ONE camera's tracks (rows) -> the vectors that are compared. Applied when a camera is
    loaded, so the track files never need redoing when the head changes. Same steps as finetune.py."""
    if len(emb) == 0:
        return emb
    if HEAD_FILE is not None and (FINETUNE_DIR / HEAD_FILE).exists():
        z = np.load(FINETUNE_DIR / HEAD_FILE)
        if "mu" in z.files:
            x = unit(emb - z["mu"])
            emb = unit(x + x @ z["W"].T)
        elif HEAD_FILE not in _warned:                      # the file of the first version (no "mu"): it collapsed, do not use it
            _warned.add(HEAD_FILE)
            print(f"{HEAD_FILE} is from the first version of the re-ID training and is ignored. Run `py finetune.py reid` to replace it.")
    if CAMERA_CENTER and len(emb) >= 5:
        emb = unit(emb - emb.mean(axis=0))
    return emb


_gate_params = []


def gate_params():
    """The gate thresholds: the built-in defaults, overridden by what finetune.py fitted (GATE_FILE)."""
    if not _gate_params:
        p, f = dict(GATE), FINETUNE_DIR / GATE_FILE
        if f.exists():
            p.update(json.loads(f.read_text()))
        _gate_params.append(p)
    return _gate_params[0]


def gate_penalties(cam, tid, other):
    """Penalty (0 plausible .. 1 impossible) of every track of `other`, in the order of other.emb_ids, against track tid of `cam`.
    None when the gate is off or one of the cameras has no calibration."""
    if not ST_GATE:
        return None
    a, b = cam.gate_tracks(), other.gate_tracks()
    if a is None or b is None:
        return None
    i = cam.emb_ids.index(tid)
    return gate_penalty(pair_quantities({k: v[i:i + 1] for k, v in a.items()}, b, p=gate_params()), gate_params())[0]


def is_mutual(cam, tid, other, other_id):
    """True when `other`'s track other_id points back at tid: tid is the best match for it among the tracks of `cam` the gate allows."""
    back = cam.match(other.emb_mat[other.emb_ids.index(other_id)], gate_penalties(other, other_id, cam))
    return back is not None and back[0] == tid


def track_camera(video, out):
    """Track the vehicles in one video. Writes `out` (frame,id,x,y,w,h,conf,-1,-1,-1, the dataset's own layout)
    and out.npz (one appearance vector per track)."""
    from ultralytics import YOLO
    model_name, tracker = BOXES.partition(":")[2].rsplit("_", 1)         # "yolo26n_bytetrack" -> yolo26n.pt + bytetrack.yaml
    weights = FINETUNE_DIR / f"{model_name}.pt"                           # fine-tuned weights live in FINETUNE_DIR, the others download by name
    model = YOLO(str(weights) if weights.exists() else f"{model_name}.pt")
    classes = [i for i, name in model.names.items() if name in VEHICLES]  # class numbers come from the model itself
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    segm = video.parent / "segm" / "segm_mask_rcnn.txt"                   # the dataset's Mask R-CNN masks of this camera
    masks = Masks(segm) if USE_MASKS and segm.exists() else None
    if USE_MASKS and masks is None:
        print(f"    no {segm}: plain crops")
    painted = [0, 0]                                                      # crops with the background removed, crops in all
    rows, best = [], {}                                                   # best: id -> its biggest crops, biggest first
    # one call over the whole video: the tracker keeps its state from frame to frame by itself
    for n, r in enumerate(model.track(str(video), tracker=f"{tracker}.yaml", classes=classes, imgsz=TRACK_IMGSZ,
                                      stream=True, verbose=False), start=1):
        if r.boxes.is_track:
            for (x1, y1, x2, y2), tid, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                rows.append(f"{n},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{conf:.2f},-1,-1,-1")
                crops, area = best.setdefault(tid, []), (x2 - x1) * (y2 - y1)
                if len(crops) < EMBED_CROPS or area > crops[-1][0]:
                    h, w = r.orig_img.shape[:2]
                    crop = r.orig_img[max(int(y1), 0):min(int(y2), h), max(int(x1), 0):min(int(x2), w)]
                    if min(crop.shape[:2]) >= 8:
                        clean = masked_crop(r.orig_img, (x1, y1, x2 - x1, y2 - y1), masks, n) if masks is not None else None
                        painted[0] += clean is not None
                        painted[1] += 1
                        crops.append((area, clean if clean is not None else crop.copy()))
                        crops.sort(key=lambda c: -c[0])
                        del crops[EMBED_CROPS:]
        if n % 100 == 0:
            print(f"    frame {n}/{total}")
    ids = sorted(i for i in best if best[i])
    owners = np.repeat(ids, [len(best[i]) for i in ids])
    vecs = embed([c for i in ids for _, c in best[i]]) if ids else np.zeros((0, 1))
    mean = np.stack([vecs[owners == i].mean(axis=0) for i in ids]) if ids else vecs
    out.parent.mkdir(parents=True, exist_ok=True)
    if masks is not None:
        print(f"    background removed on {painted[0]} of {painted[1]} crops")
    np.savez(out.with_suffix(".npz"), ids=np.array(ids, dtype=int), masked=np.array(masks is not None),
             emb=mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-9))
    out.write_text("\n".join(rows) + "\n")                                # written last: it marks the camera as done


def make_tracks(folder):
    """Track every camera of the scenario that has no track file yet, one after another."""
    todo = [p for p in sorted(folder.iterdir()) if (p / "vdo.avi").exists() and not boxes_path(p).exists()]
    for k, p in enumerate(todo, start=1):
        print(f"Tracking {folder.name}/{p.name}  ({k} of {len(todo)}) ...")
        track_camera(p / "vdo.avi", boxes_path(p))


# ---- the map ---------------------------------------------------------------------------------------------------------
def tile_coordinates(lat, lon, z):
    """Where (lat, lon) is on the Web Mercator tile grid of zoom z, in tiles: the whole part is the tile number."""
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n


def tile_path(z, x, y):
    """Where tile (z, x, y) is kept; one folder per source, so changing SATELLITE never mixes photos."""
    source = f"{urllib.parse.urlparse(SATELLITE).netloc}_{zlib.crc32(SATELLITE.encode()):08x}"
    return TILES_DIR / source / str(z) / f"{x}_{y}.jpg"


def satellite_tile(z, x, y):
    """One 256 x 256 satellite tile as an image, kept on disk after the first download. None if it can't be had."""
    path = tile_path(z, x, y)
    if path.exists():
        return cv2.imread(str(path))
    try:
        request = urllib.request.Request(SATELLITE.format(z=z, x=x, y=y),
                                         headers={"User-Agent": "multicam-cctv-viewer (personal portfolio project)"})
        with urllib.request.urlopen(request, timeout=20) as reply:
            data = reply.read()
    except Exception:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != (256, 256):         # an error page, not a tile
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return img


def ground_footprint(calib, roi):
    """The road a camera sees, as a polygon of (lat, lon) points: its road mask, without the sky and without everything so far
    away that one pixel covers more than FOOTPRINT_RES metres of ground. None if nothing is left."""
    h, w = roi.shape
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    pts = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
    pts = pts[roi[pts[:, 1].astype(int), pts[:, 0].astype(int)]]

    def metres(p):                                   # east, north of this camera's own centre, and the horizon check
        lat, lon, ok = calib.pixels_to_gps(p)
        return np.c_[(lon - calib.lon0) * calib.m_per_deg_lon, (lat - calib.lat0) * calib.m_per_deg_lat], ok

    here, ok1 = metres(pts)
    right, ok2 = metres(pts + [1, 0])
    below, ok3 = metres(pts + [0, 1])
    res = np.maximum(np.hypot(*(right - here).T), np.hypot(*(below - here).T))      # metres of ground per pixel
    keep = ok1 & ok2 & ok3 & (res <= FOOTPRINT_RES)
    if keep.sum() < 3:
        return None
    hull = cv2.convexHull(here[keep].astype(np.float32))[:, 0, :]
    return np.c_[calib.lat0 + hull[:, 1] / calib.m_per_deg_lat, calib.lon0 + hull[:, 0] / calib.m_per_deg_lon]


# ---- the player ---------------------------------------------------------------------------------------------------
class Clock:
    """Scenario time. It follows the wall clock while playing, so playback runs at real-world speed."""

    def __init__(self, start, end):
        self.start, self.end = start, end
        self.t, self.playing, self._ref = start, True, time.perf_counter()

    def now(self):
        if not self.playing:
            return self.t
        t = self.t + time.perf_counter() - self._ref
        if t >= self.end:                                  # reached the end: stop there
            self.seek(self.end)
            self.playing = False
            return self.end
        return t

    def seek(self, t):
        self.t, self._ref = min(max(t, self.start), self.end), time.perf_counter()

    def toggle(self):
        t = self.now()
        self.seek(self.start if (not self.playing and t >= self.end) else t)   # play from the end -> start over
        self.playing = not self.playing


class Cam:
    def __init__(self, name, folder, offset):
        self.name, self.offset = name, offset
        self.fps = FPS.get(name, DEFAULT_FPS)
        self.cap = cv2.VideoCapture(str(folder / "vdo.avi"))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        path = boxes_path(folder)
        self.has_boxes = path.exists()
        self.boxes = load_mot(path) if self.has_boxes else {}
        self.emb_ids, self.emb_mat = [], np.zeros((0, 1))      # appearance vector of every track (BOXES = "tracks:...")
        if self.has_boxes and path.with_suffix(".npz").exists():
            z = np.load(path.with_suffix(".npz"))
            self.emb_ids, self.emb_mat = z["ids"].tolist(), reid_transform(z["emb"])
            if (bool(z["masked"]) if "masked" in z.files else False) != USE_MASKS:
                print(f"{name}: its track file was made {'without' if USE_MASKS else 'with'} background removal but USE_MASKS = {USE_MASKS}. "
                      f"Delete the folder {path.parent} to track this scenario again, or change USE_MASKS, so the vectors match the re-ID head.")
        self.calib, self.footprint = None, None                # for the map: where this camera is and which road it sees
        if (folder / "calibration.txt").exists():
            roi = load_roi(folder / "roi.jpg") if (folder / "roi.jpg").exists() else None
            w, h = (roi.shape[1], roi.shape[0]) if roi is not None else (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080)
            self.calib = Calibration(folder / "calibration.txt", (w, h))
            self.footprint = ground_footprint(self.calib, roi if roi is not None else np.ones((h, w), bool))
        self._gate = None              # spatio-temporal summary of every track with an appearance vector, made when first needed
        self.target = None             # None: show every box. Otherwise {id: similarity or None}: show only these, in red
        self.hits = []                 # (id, x1, y1, x2, y2) of the boxes drawn in the tile, for clicking
        self.pos, self.frame = -1, None                    # index of the frame held in self.frame
        self.tile, self.tile_key = None, None

    @property
    def end(self):
        return self.offset + self.n / self.fps

    def track_points(self, tid):
        """(scenario time, lat, lon) of the bottom-centre of every box of track tid, earliest first. None without calibration."""
        if self.calib is None:
            return None
        rows = [(f, r[1] + r[3] / 2, r[2] + r[4]) for f, rs in self.boxes.items() for r in rs if int(r[0]) == tid]
        if not rows:
            return None
        f, x, y = np.array(sorted(rows)).T
        lat, lon, ok = self.calib.pixels_to_gps(np.c_[x, y])
        return self.offset + (f[ok] - 1) / self.fps, lat[ok], lon[ok]       # frames count from 1

    def gate_tracks(self):
        """When and where each track (in the order of emb_ids) enters and leaves this camera, as arrays for the gate. None if this
        camera has no calibration or no tracks."""
        if self._gate is None:
            self._gate = False
            if self.calib is not None and self.emb_ids and self.boxes:
                rows = np.vstack([np.c_[np.full(len(r), f), r[:, :5]] for f, r in self.boxes.items() if len(r)])    # frame, id, x, y, w, h
                lat, lon, ok = self.calib.pixels_to_gps(np.c_[rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5]])
                xy = np.c_[to_metres(lat, lon, SCENARIO_CENTER[SCENARIO])]
                t = self.offset + (rows[:, 0] - 1) / self.fps                                                      # frames count from 1
                tracks = []
                for tid in self.emb_ids:
                    m = (rows[:, 1] == tid) & ok
                    order = np.argsort(t[m])
                    tracks.append(summarize_track(t[m][order], xy[m][order]))
                self._gate = stack_tracks(tracks)
        return self._gate or None

    def set_target(self, target):
        self.target, self.tile_key = target, None          # the tile has to be drawn again

    def match(self, vec, pen=None):
        """The track of this camera that looks most like vec: (id, similarity), or None if it has no appearance vectors or the gate
        (pen: penalty per track, see gate_penalties) rules out all of them."""
        if not self.emb_ids:
            return None
        sims = self.emb_mat @ vec
        if pen is None:
            k = int(np.argmax(sims))
        else:
            score = np.where(pen >= 1.0, -np.inf, sims - ST_WEIGHT * pen)
            k = int(np.argmax(score))
            if not np.isfinite(score[k]):
                return None
        return self.emb_ids[k], float(sims[k])

    def index_at(self, t):
        """Index of the frame on screen at scenario time t, or None when this camera isn't recording then."""
        i = math.floor((t - self.offset) * self.fps + 1e-6)
        return i if 0 <= i < self.n else None

    def read(self, i):
        """Frame i (counting from 0). Plays forward when it can, dropping frames if it is behind."""
        if i == self.pos:
            return self.frame
        if self.pos < i <= self.pos + MAX_SKIP:
            for _ in range(i - self.pos - 1):
                self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        self.pos, self.frame = i, (frame if ok else None)
        return self.frame

    def render(self, t, size, show_boxes):
        """The tile for scenario time t. Reused until the frame, the size or the box setting changes."""
        i = self.index_at(t)
        if i is not None and (i, size, show_boxes) == self.tile_key:
            return self.tile
        tw, th = size
        tile = np.zeros((th, tw, 3), np.uint8)
        self.hits = []
        frame = self.read(i) if i is not None else None
        if frame is None:
            msg = f"starts at {fmt(self.offset)}" if t < self.offset else "ended" if i is None else "no frame"
            put_text(tile, f"{self.name}  {msg}", (10, 26), 0.6, (180, 180, 180))
            return tile
        h, w = frame.shape[:2]
        s = min(tw / w, th / h)
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        x0, y0 = (tw - small.shape[1]) // 2, (th - small.shape[0]) // 2
        tile[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        if show_boxes:
            for tid, x, y, bw, bh, _ in self.boxes.get(i + 1, EMPTY):          # the dataset counts frames from 1
                tid = int(tid)
                if self.target is not None and tid not in self.target:
                    continue                                                    # following someone else: hide the rest
                p1 = (x0 + int(x * s), y0 + int(y * s))
                p2 = (x0 + int((x + bw) * s), y0 + int((y + bh) * s))
                if self.target is None:
                    color, thick, label = BOX_COLOR, 2, f"#{tid}"
                else:
                    sim = self.target[tid]
                    color, thick, label = TARGET_COLOR, 3, "TARGET" if sim is None else f"TARGET {sim:.2f}"
                cv2.rectangle(tile, p1, p2, color, thick)
                put_text(tile, label, (p1[0], max(p1[1] - 5, 14)), 0.5, color if self.target is None else (255, 255, 255))
                self.hits.append((tid, p1[0], p1[1], p2[0], p2[1]))
        put_text(tile, f"{self.name}  {fmt(self.offset + i / self.fps)}", (10, 26), 0.6)
        self.tile, self.tile_key = tile, (i, size, show_boxes)
        return tile


class Player:
    def __init__(self, cams, clock):
        self.cams, self.clock = cams, clock
        self.show_boxes, self.dragging = SHOW_BOXES, False
        self.path = []                                     # the followed vehicle on the map: [(times, east, north), ...]
        self.setup_map()

    def to_xy(self, lat, lon):
        return np.c_[(np.asarray(lon) - self.lon0) * self.mlon, (np.asarray(lat) - self.lat0) * self.mlat]

    def px(self, xy):
        """Metres east/north -> pixels of the map panel (north is up)."""
        x = WINDOW[0] / 2 + (xy[:, 0] - self.cx) * self.scale
        y = MAP_H / 2 - (xy[:, 1] - self.cy) * self.scale
        return np.clip(np.c_[x, y].round(), -30000, 30000).astype(np.int32)

    def satellite_photo(self):
        """The satellite picture behind the map, or None. For every pixel of the panel the place on Earth is worked out and its
        colour looked up in the tiles, so the photo lines up exactly with the footprints and paths drawn on it."""
        if SATELLITE is None:
            return None
        W = WINDOW[0]
        xs, ys = np.meshgrid(np.arange(W), np.arange(MAP_H))
        lat = self.lat0 + (self.cy - (ys - MAP_H / 2) / self.scale) / self.mlat
        lon = self.lon0 + (self.cx + (xs - W / 2) / self.scale) / self.mlon
        # the lowest zoom whose tiles are about as sharp as the panel (up to 20% coarser looks the same and downloads a quarter
        # of the tiles), but no sharper than the source has
        z = min(SATELLITE_MAX_ZOOM, math.ceil(math.log2(156543.03392 * math.cos(math.radians(self.lat0)) * self.scale) - 0.25))
        while True:
            tx, ty = tile_coordinates(lat, lon, z)
            x0, x1, y0, y1 = int(tx.min()), int(tx.max()), int(ty.min()), int(ty.max())
            if (x1 - x0 + 1) * (y1 - y0 + 1) <= SATELLITE_MAX_TILES or z <= 0:
                break
            z -= 1                                         # too many tiles for this view: fewer, softer ones
        wanted = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        if not all(tile_path(z, x, y).exists() for x, y in wanted):
            print(f"Getting {len(wanted)} satellite tiles for the map (first time only, then kept in {TILES_DIR}) ...")
        mosaic = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), 38, np.uint8)
        got = missed = 0
        for x, y in wanted:
            img = satellite_tile(z, x, y)
            if img is None:
                missed += 1
                if missed >= 3 and got == 0:               # not reachable at all: don't wait for the rest
                    break
            else:
                mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = img
                got += 1
        if got == 0:
            print(f"Couldn't get satellite tiles from {urllib.parse.urlparse(SATELLITE).netloc}; the map is drawn without a photo.")
            return None
        if missed:
            print(f"{missed} of {len(wanted)} satellite tiles could not be had; those parts of the map stay dark.")
        photo = cv2.remap(mosaic, ((tx - x0) * 256).astype(np.float32), ((ty - y0) * 256).astype(np.float32), cv2.INTER_LINEAR)
        photo = (photo * 0.8).astype(np.uint8)               # a little darker, so labels and paths stand out
        put_text(photo, SATELLITE_CREDIT, (W - 7 * len(SATELLITE_CREDIT) - 14, MAP_H - 8), 0.4, (200, 200, 200))
        return photo

    def setup_map(self):
        """Everything on the map that never changes: camera footprints, scale bar, north arrow. Also fixes the view."""
        W = WINDOW[0]
        self.map_img = np.full((MAP_H, W, 3), 38, np.uint8)
        shown = [(c, c.footprint) for c in self.cams if c.footprint is not None]
        self.has_map = bool(shown)
        if not shown:
            put_text(self.map_img, "no calibration.txt found: nothing to show on the map", (10, 26), 0.6, (180, 180, 180))
            return
        self.lat0, self.lon0 = np.vstack([fp for _, fp in shown]).mean(axis=0)
        self.mlat, self.mlon = 111132.0, 111320.0 * math.cos(math.radians(self.lat0))
        shapes = [self.to_xy(fp[:, 0], fp[:, 1]) for _, fp in shown]
        lo, hi = np.vstack(shapes).min(axis=0), np.vstack(shapes).max(axis=0)
        span = np.maximum(hi - lo, 20.0)                   # at least 20 m, so a lone small footprint isn't blown up
        self.cx, self.cy = (lo + hi) / 2
        self.scale = min((W - 60) / span[0], (MAP_H - 60) / span[1])
        photo = self.satellite_photo()
        if photo is not None:
            self.map_img = photo
        for k, ((cam, _), xy) in enumerate(zip(shown, shapes)):
            pts, color = self.px(xy), MAP_COLORS[self.cams.index(cam) % len(MAP_COLORS)]
            layer = self.map_img.copy()
            cv2.fillPoly(layer, [pts], color)
            self.map_img = cv2.addWeighted(layer, 0.3, self.map_img, 0.7, 0)
            cv2.polylines(self.map_img, [pts], True, color, 1, cv2.LINE_AA)
            cx, cy = (int(v) for v in pts.mean(axis=0))                      # a marker too: far apart, a footprint is a speck
            cv2.circle(self.map_img, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(self.map_img, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
            put_text(self.map_img, cam.name, (cx - 18, cy - 12), 0.45)
        nice = max([n for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if n * self.scale <= 160] or [10])
        y = MAP_H - 14
        cv2.line(self.map_img, (14, y), (14 + int(nice * self.scale), y), (255, 255, 255), 2)
        put_text(self.map_img, f"{nice} m" if nice < 1000 else f"{nice // 1000} km", (14, y - 8), 0.45)
        cv2.arrowedLine(self.map_img, (W - 30, 50), (W - 30, 18), (255, 255, 255), 2, tipLength=0.35)
        put_text(self.map_img, "N", (W - 38, 70), 0.45)

    def target_path(self):
        """The followed vehicle on the map: one piece for every camera that sees it, as (times, east, north), earliest first."""
        pieces = []
        for cam in self.cams:
            for tid in cam.target or {}:
                got = cam.track_points(tid) if self.has_map else None
                if got is not None and len(got[0]):
                    pieces.append((got[0], self.to_xy(got[1], got[2])))
        return sorted(pieces, key=lambda p: p[0][0])

    def map_panel(self, t):
        """The map at scenario time t: the fixed part plus the followed vehicle's path so far."""
        img = self.map_img.copy()
        last, prev_end = None, None
        for times, xy in self.path:
            pts = self.px(xy)
            cv2.polylines(img, [pts], False, DARK_RED, 2, cv2.LINE_AA)                       # the whole path, faint
            if prev_end is not None:
                cv2.line(img, prev_end, (int(pts[0][0]), int(pts[0][1])), DARK_RED, 1, cv2.LINE_AA)   # not seen in between
            prev_end = (int(pts[-1][0]), int(pts[-1][1]))
            done = times <= t
            if done.any():
                cv2.polylines(img, [pts[done]], False, (0, 0, 0), 5, cv2.LINE_AA)              # outline, so it reads on a photo
                cv2.polylines(img, [pts[done]], False, TARGET_COLOR, 3, cv2.LINE_AA)           # the part already driven
                if last is None or times[done][-1] >= last[0]:
                    last = (times[done][-1], (int(pts[done][-1][0]), int(pts[done][-1][1])))
        if last is not None:
            cv2.circle(img, last[1], 7, (255, 255, 255), -1)
            cv2.circle(img, last[1], 5, TARGET_COLOR, -1)
            put_text(img, "TARGET", (last[1][0] + 10, last[1][1] + 20), 0.5)
        return img

    @staticmethod
    def bar_x():
        return 300, WINDOW[0] - 20                          # where the track starts and ends

    def grid(self):
        """(columns, (tile width, tile height))"""
        n = len(self.cams)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, (WINDOW[0] // cols, (WINDOW[1] - BAR_H - MAP_H) // rows)

    def render(self):
        """One picture of the whole window. The clock is read once, so every tile shows the same moment."""
        W, H = WINDOW
        t = self.clock.now()
        canvas = np.zeros((H, W, 3), np.uint8)
        cols, (tw, th) = self.grid()
        for k, cam in enumerate(self.cams):
            r, c = divmod(k, cols)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cam.render(t, (tw, th), self.show_boxes)
            cv2.rectangle(canvas, (c * tw, r * th), ((c + 1) * tw - 1, (r + 1) * th - 1), (60, 60, 60), 1)
        canvas[H - BAR_H - MAP_H:H - BAR_H] = self.map_panel(t)
        x0, x1 = self.bar_x()
        y = H - BAR_H // 2
        canvas[H - BAR_H:] = 30
        state = "PLAYING" if self.clock.playing else "PAUSED"
        put_text(canvas, f"{state}  {fmt(t)} / {fmt(self.clock.end)}", (10, y + 6), 0.55)
        span = max(self.clock.end - self.clock.start, 1e-9)
        xk = int(x0 + (t - self.clock.start) / span * (x1 - x0))
        cv2.line(canvas, (x0, y), (x1, y), (90, 90, 90), 4)
        cv2.line(canvas, (x0, y), (xk, y), BAR_COLOR, 4)
        cv2.circle(canvas, (xk, y), 8, (255, 255, 255), -1)
        return canvas

    def follow(self, cam, tid):
        """Follow vehicle tid of `cam`: red box on it, nothing else drawn, and the same vehicle looked for in the other cameras."""
        for c in self.cams:
            c.set_target({})                                # following: draw only what is found below
        cam.set_target({tid: None})
        if BOXES == "gt":                                   # the dataset's ids are the same in every camera
            for c in self.cams:
                c.set_target({tid: None})
        elif tid in cam.emb_ids:                            # otherwise: the most similar track of every other camera
            vec = cam.emb_mat[cam.emb_ids.index(tid)]
            notes = []
            for c in self.cams:
                if c is cam:
                    continue
                found = c.match(vec, gate_penalties(cam, tid, c))
                if found is None:
                    notes.append(f"{c.name}: nothing plausible")
                    continue
                shown = found[1] >= REID_MIN and (not RECIPROCAL or is_mutual(cam, tid, c, found[0]))
                if shown:
                    c.set_target({found[0]: found[1]})
                notes.append(f"{c.name}: track {found[0]} ({found[1]:.2f})" + ("" if shown else ", hidden"))
            print(f"follow {tid} of {cam.name} -> " + "; ".join(notes))
        self.path = self.target_path()

    def stop_following(self):
        for c in self.cams:
            c.set_target(None)
        self.path = []

    def key(self, k):
        """Handle one key press. Returns False when it is time to quit."""
        ch = chr(k).lower() if 0 <= k < 256 else ""
        if k == 27 or ch == "q":
            return False
        if k == 32 or ch == "k":
            self.clock.toggle()
        elif ch == "a" or k in LEFT:
            self.clock.seek(self.clock.now() - JUMP)
        elif ch == "d" or k in RIGHT:
            self.clock.seek(self.clock.now() + JUMP)
        elif ch == "r":
            self.clock.seek(self.clock.start)
        elif ch == "b":
            self.show_boxes = not self.show_boxes
        elif ch == "c":
            self.stop_following()
        return True

    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and y >= WINDOW[1] - BAR_H:
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        if self.dragging and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            x0, x1 = self.bar_x()
            frac = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
            self.clock.seek(self.clock.start + frac * (self.clock.end - self.clock.start))
        elif event == cv2.EVENT_LBUTTONDOWN and y < WINDOW[1] - BAR_H - MAP_H:   # a click on a video tile: pick a vehicle
            cols, (tw, th) = self.grid()
            k = (y // th) * cols + min(x // tw, cols - 1)
            if k < len(self.cams):
                cam, px, py = self.cams[k], x - min(x // tw, cols - 1) * tw, y % th
                hit = [h for h in cam.hits if h[1] <= px <= h[3] and h[2] <= py <= h[4]]
                if hit:                                                      # the smallest box under the click
                    self.follow(cam, min(hit, key=lambda h: (h[3] - h[1]) * (h[4] - h[2]))[0])


def read_offsets():
    path = OTHERS / "cam_timestamp" / f"{SCENARIO}.txt"
    if not path.exists():
        raise SystemExit(f"Can't find {path}\nSet OTHERS at the top of this file to the folder that contains cam_timestamp/.")
    return {name: float(sec) for name, sec in (line.split() for line in path.read_text().splitlines() if line.strip())}


def build_player():
    offsets = read_offsets()
    folder = ROOT / SPLIT / SCENARIO
    if BOXES.startswith("tracks:"):
        make_tracks(folder)
    wanted = CAMERAS or sorted(p.name for p in folder.iterdir() if p.is_dir())
    names = [n for n in wanted if n in offsets and (folder / n / "vdo.avi").exists()][:MAX_TILES]
    if not names:
        raise SystemExit(f"No cameras found in {folder} (with a vdo.avi and an entry in cam_timestamp).")
    cams = [Cam(n, folder / n, offsets[n]) for n in names]
    missing = [c.name for c in cams if not c.has_boxes]
    if missing:
        print(f'No boxes for {", ".join(missing)}: no file for BOXES = "{BOXES}". '
              f'Try "tracks:yolo26n_bytetrack" (made for you) or "mtsc:tc_mask_rcnn" (any mtsc_<name>.txt in the mtsc folder).')
    no_map = [c.name for c in cams if c.calib is None]
    if no_map:
        print(f"No calibration.txt for {', '.join(no_map)}: they won't appear on the map.")
    clock = Clock(min(c.offset for c in cams), max(c.end for c in cams))
    print(f"{SCENARIO}: {', '.join(names)}   timeline {fmt(clock.start)} to {fmt(clock.end)}")
    return Player(cams, clock)


def main():
    player = build_player()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, player.mouse)
    while True:
        t0 = time.perf_counter()
        cv2.imshow(WIN, player.render())
        k = cv2.waitKeyEx(max(1, 33 - int((time.perf_counter() - t0) * 1000)))      # about 30 redraws a second
        if (k != -1 and not player.key(k)) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
