"""finetune.py: fine-tuning for the detector and for re-identification, all in this one file. It REPLACES the earlier ft_*.py scripts.

  py finetune.py det [model]   detector (default yolo26n; try yolo26s, yolo26m): build the dataset, train, then compare on S02+S05.
                               Each step is skipped when its result exists (delete the file to redo it). The dataset is built once
                               and shared by every model size. Weights: cityflow_yolo.pt (n) / cityflow_<model>.pt in WORK.
  py finetune.py reid          re-ID + spatio-temporal gate: train on S01+S03+S04, score on S02+S05, write reid_head.npz and gate.json,
                               and print the settings for multicam.py.
  py finetune.py reid-all      the same trained on all 5 labelled scenarios (what the top AI City team did) -> reid_head_all.npz, gate_all.json.

Everything it creates lives inside WORK (datasets, caches, weights, heads), nothing is left next to your scripts.
Needs cityflow.py (the updated one) next to it. multicam.py reads the weights, head and gate from WORK (see its settings).
Work already done by the earlier scripts is reused: the detector weights are moved into WORK and the re-ID crop features stay valid.

Detector note: gt.txt only labels vehicles seen by 2+ cameras, so the original model's confident detections of the other vehicles
are added as extra labels (PSEUDO). Otherwise the detector would learn that those vehicles are background.

Re-ID notes: CLIP stays frozen and a small head is trained on its cached vectors:
    z = normalise(x - mu)  ->  out = normalise(z + W z)      (W starts at 0 = plain centred CLIP)
with a supervised-contrastive loss whose positives are the SAME vehicle in a DIFFERENT camera. Training examples are tracklet vectors
(mean of 5 crops), exactly what multicam.py stores per track. With USE_MASKS the background of every crop is painted grey using the
dataset's Mask R-CNN masks, so the vector describes the vehicle and not the road behind it (multicam.py must use the same setting).
The spatio-temporal gate (see cityflow.py) removes matches that physics rules out; its thresholds are fitted to the labelled scenarios.
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cityflow import (DEFAULT_FPS, FPS, GATE, SCENARIO_CENTER, Calibration, Masks, gate_penalty, iou_matrix, load_mot, load_roi,
                      masked_crop, pair_quantities, roi_fraction, stack_tracks, summarize_track, to_metres)

# ============================================================================================================================
# Settings
# ============================================================================================================================
ROOT = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\AICity22_Track1_MTMC_Tracking")
OTHERS = ROOT                                                          # the folder that contains cam_timestamp/ (same as OTHERS in multicam.py)
WORK = Path(r"C:\Users\BaneH\Desktop\AI CV PROJECT\finetune")        # datasets, caches, weights, heads
TRAIN = [("train", "S01"), ("train", "S03"), ("train", "S04")]
VAL = [("validation", "S02"), ("validation", "S05")]                 # never trained on by `det` / `reid`: only used for the scores

VEHICLES = ("car", "motorcycle", "bus", "truck")                       # same filter multicam.py uses
CLIP_MODEL = "openai/clip-vit-base-patch32"                            # same model multicam.py uses
EMBED_CROPS = 5                                                        # same as multicam.py: biggest crops per track

# detector
DEFAULT_MODEL = "yolo26n"           # `py finetune.py det` without a model name
PSEUDO_BASE = "yolo26n.pt"          # the model that finds the unlabeled vehicles used as extra labels (see PSEUDO)
DET_WEIGHTS = WORK / "cityflow_yolo.pt"                                # weights of yolo26n; other sizes: cityflow_<model>.pt
DET_DATA = WORK / "det_dataset"
STEP = 30                           # keep a GT frame only if frame % STEP == 0 (30 = one frame every 3 s at 10 FPS)
MAX_PER_CAM = 100                   # and at most this many frames per camera, evenly spread
PSEUDO = True
PSEUDO_CONF = 0.5                   # only confident original detections become extra labels
PSEUDO_MAX_IOU = 0.3                # ...and only when they do not overlap a GT box
PSEUDO_IN_ROI = 0.5                 # ...and lie mostly inside the camera's ROI (the dataset's own definition of "annotated area")
PSEUDO_IMGSZ = 960
EPOCHS, IMGSZ, BATCH, FREEZE = 6, 640, 8, 10     # FREEZE: first 10 layers (backbone) stay fixed: faster and safer on CPU
EVAL_FRAMES_PER_CAM = 30
EVAL_IMGSZ = 1280                   # what multicam.py tracks at (TRACK_IMGSZ); lower it if eval is too slow
EVAL_CONF = 0.25

# re-id
CROPS_PER_TRACK = 24        # crops kept per (camera, vehicle) when extracting features (changing it means deleting the caches)
MIN_SIDE = 24               # ignore crops smaller than this (pixels)
USE_MASKS = True            # paint the background of every crop grey (Mask R-CNN masks). multicam.py's USE_MASKS must match
ST_WEIGHTS = (0.0, 0.25, 0.5)   # penalty weights tried for the soft part of the gate (0 = only reject the impossible)
P, K = 24, 4                # a batch = P vehicles of one scenario x K tracklets (K/2 in each of two different cameras)
TEMP = 0.07                 # contrastive temperature: smaller = punishes confusable pairs harder
ITERS = 2000
LR, WD, DROPOUT = 1e-3, 5e-2, 0.2
TARGET_PRECISION = 0.6      # the settings printed at the end aim for this share of shown matches being right
SEED = 0
HEAD_OUT, HEAD_OUT_ALL = WORK / "reid_head.npz", WORK / "reid_head_all.npz"
GATE_OUT, GATE_OUT_ALL = WORK / "gate.json", WORK / "gate_all.json"
EXCLUDED = -10.0            # score of a candidate the gate rules out


# ============================================================================================================================
# Helpers
# ============================================================================================================================
def cameras(split, scenario):
    """Camera folders of a scenario that have a video and ground truth."""
    folder = ROOT / split / scenario
    return [p for p in sorted(folder.iterdir()) if (p / "vdo.avi").exists() and (p / "gt" / "gt.txt").exists()]


def read_frames(video, wanted):
    """Yield (index from 0, frame) for the wanted indices, in order. Skips the rest with grab(), which is much cheaper."""
    wanted = set(wanted)
    if not wanted:
        return
    cap, i, last = cv2.VideoCapture(str(video)), 0, max(wanted)
    while i <= last:
        if i in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
        elif not cap.grab():
            break
        i += 1
    cap.release()


def even_pick(items, n):
    """At most n items, evenly spread over the list."""
    if len(items) <= n:
        return list(items)
    return [items[j] for j in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


_clip = []


def clip_embed(crops_bgr):
    """One unit-length CLIP vector per image. Identical to embed() in multicam.py, so training and viewer agree."""
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


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


# ============================================================================================================================
# Detector
# ============================================================================================================================
def weights_for(model):
    return DET_WEIGHTS if model == DEFAULT_MODEL else WORK / f"cityflow_{model}.pt"


def adopt_old_weights():
    """Weights from the first version of these scripts sat next to multicam.py: bring them into WORK so nothing is retrained."""
    if DET_WEIGHTS.exists():
        return
    for old, move in ((Path("cityflow_yolo.pt"), True), (WORK / "runs" / "det" / "weights" / "last.pt", False)):
        if old.exists():
            WORK.mkdir(parents=True, exist_ok=True)
            (shutil.move if move else shutil.copy)(str(old), str(DET_WEIGHTS))
            print(f"found earlier weights {old} -> {DET_WEIGHTS}")
            return


def yolo_lines(boxes_xywh, W, H):
    lines = []
    for x, y, w, h in boxes_xywh:
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def det_build():
    from ultralytics import YOLO
    base = YOLO(PSEUDO_BASE) if PSEUDO else None
    vclasses = [i for i, n in base.names.items() if n in VEHICLES] if PSEUDO else []
    img_dir, lab_dir = DET_DATA / "images" / "train", DET_DATA / "labels" / "train"
    shutil.rmtree(DET_DATA, ignore_errors=True)
    img_dir.mkdir(parents=True), lab_dir.mkdir(parents=True)
    total = extra = 0
    for split, scn in TRAIN:
        for cam in cameras(split, scn):
            gt = load_mot(cam / "gt" / "gt.txt")
            frames = even_pick(sorted(f for f in gt if f % STEP == 0), MAX_PER_CAM)
            roi = load_roi(cam / "roi.jpg") if (cam / "roi.jpg").exists() else None
            print(f"{scn}/{cam.name}: {len(frames)} frames")
            for i, frame in read_frames(cam / "vdo.avi", [f - 1 for f in frames]):
                H, W = frame.shape[:2]
                boxes = [tuple(r) for r in gt[i + 1][:, 1:5]]
                if PSEUDO:
                    r = base.predict(frame, classes=vclasses, conf=PSEUDO_CONF, imgsz=PSEUDO_IMGSZ, device="cpu", verbose=False)[0]
                    det = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]).reshape(-1, 4)
                    if len(det):
                        free = iou_matrix(det, gt[i + 1][:, 1:5]).max(axis=1) < PSEUDO_MAX_IOU
                        for d, ok in zip(det, free):
                            if ok and (roi is None or roi_fraction(roi, *d) >= PSEUDO_IN_ROI):
                                boxes.append(tuple(d))
                                extra += 1
                name = f"{scn}_{cam.name}_{i + 1:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (lab_dir / f"{name}.txt").write_text("\n".join(yolo_lines(boxes, W, H)) + "\n")
                total += 1
    (DET_DATA / "data.yaml").write_text(                              # written last: it marks the dataset as complete
        f"path: {DET_DATA.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: car\n")     # val unused (val=False below)
    print(f"{total} images written, {extra} pseudo-labels added -> {DET_DATA}")


def det_train(model):
    from ultralytics import YOLO
    net = YOLO(f"{model}.pt")
    net.train(data=str(DET_DATA / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device="cpu", workers=2,
              freeze=FREEZE, optimizer="AdamW", lr0=5e-4, warmup_epochs=1, val=False,
              project=str(WORK / "runs"), name=f"det_{model}", exist_ok=True, plots=False)
    out = weights_for(model)
    shutil.copy(Path(net.trainer.save_dir) / "weights" / "last.pt", out)
    print(f"saved {out}   ->   BOXES = \"tracks:{out.stem}_bytetrack\"")


def det_eval(model):
    """Recall of the GT boxes on validation scenarios (the GT is incomplete, so precision/mAP would punish correct detections
    of unlabeled vehicles; recall of labeled vehicles is the fair number). Also compares with every other fine-tuned model in WORK."""
    from ultralytics import YOLO
    mine = weights_for(model)
    models = {f"{model} original": YOLO(f"{model}.pt"), f"{model} fine-tuned": YOLO(str(mine))}
    for p in sorted(WORK.glob("cityflow_*.pt")):
        if p != mine:
            models[p.stem] = YOLO(str(p))
    stats = {k: {"gt": 0, "r50": 0, "r70": 0, "dup": 0, "dets": 0, "frames": 0} for k in models}
    for split, scn in VAL:
        for cam in cameras(split, scn):
            gt = load_mot(cam / "gt" / "gt.txt")
            frames = even_pick(sorted(gt), EVAL_FRAMES_PER_CAM)
            print(f"{scn}/{cam.name}: {len(frames)} frames")
            for i, frame in read_frames(cam / "vdo.avi", [f - 1 for f in frames]):
                g = gt[i + 1][:, 1:5]
                for name, m in models.items():
                    classes = [k for k, n in m.names.items() if n in VEHICLES]
                    r = m.predict(frame, classes=classes, conf=EVAL_CONF, imgsz=EVAL_IMGSZ, device="cpu", verbose=False)[0]
                    d = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]).reshape(-1, 4)
                    iou = iou_matrix(g, d) if len(d) else np.zeros((len(g), 0))
                    best = iou.max(axis=1) if len(d) else np.zeros(len(g))
                    s = stats[name]
                    s["gt"] += len(g); s["r50"] += int((best >= 0.5).sum()); s["r70"] += int((best >= 0.7).sum())
                    s["dup"] += int(((iou >= 0.5).sum(axis=1) >= 2).sum())          # one labeled vehicle boxed 2+ times
                    s["dets"] += len(d); s["frames"] += 1
    print("\nValidation scenarios (S02, S05), GT vehicles found. duplicates = share of GT vehicles boxed 2+ times at IoU>=0.5:")
    for name, s in stats.items():
        print(f"  {name:24s} recall@IoU0.5 {s['r50'] / s['gt']:.3f}   recall@IoU0.7 {s['r70'] / s['gt']:.3f}   "
              f"duplicates {s['dup'] / s['gt']:.1%}   detections/frame {s['dets'] / s['frames']:.1f}   ({s['gt']} GT boxes)")


def detector(model=DEFAULT_MODEL):
    WORK.mkdir(parents=True, exist_ok=True)
    adopt_old_weights()
    if not weights_for(model).exists() and not (DET_DATA / "data.yaml").exists():
        det_build()
    if not weights_for(model).exists():
        det_train(model)
    det_eval(model)


# ============================================================================================================================
# Re-identification: features of GT crops (cached in WORK, delete a cache to redo it)
# ============================================================================================================================
def collect(scenarios, name, masked=False):
    """CLIP vector of GT crops -> {feats, labels, cams, areas}. With masked=True the background of every crop is painted grey
    (Mask R-CNN masks of the dataset); the crops chosen are exactly the same, so the two caches can be compared row by row."""
    cache = WORK / f"reid_{name}{'_masked' if masked else ''}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    feats, labels, cams, areas = [], [], [], []
    used = total = 0
    for split, scn in scenarios:
        for cam in cameras(split, scn):
            gt = load_mot(cam / "gt" / "gt.txt")
            per = defaultdict(list)                                    # id -> [(frame, x, y, w, h)]
            for f, rows in gt.items():
                for r in rows:
                    if min(r[3], r[4]) >= MIN_SIDE:
                        per[int(r[0])].append((f, *r[1:5]))
            picks = defaultdict(list)                                  # frame -> [(id, x, y, w, h)]
            for tid, lst in per.items():
                for f, *box in even_pick(sorted(lst), CROPS_PER_TRACK):
                    picks[f].append((tid, *box))
            masks, segm = None, cam / "segm" / "segm_mask_rcnn.txt"
            if masked:
                if segm.exists():
                    masks = Masks(segm)
                else:
                    print(f"    {scn}/{cam.name}: no {segm.name}, plain crops for this camera")
            crops, meta, n_masked = [], [], 0
            for i, frame in read_frames(cam / "vdo.avi", [f - 1 for f in picks]):
                H, W = frame.shape[:2]
                for tid, x, y, w, h in picks[i + 1]:
                    c = masked_crop(frame, (x, y, w, h), masks, i + 1) if masks is not None else None
                    if c is not None:
                        n_masked += 1
                    else:                                              # no mask found (or plain mode): the box as it is
                        c = frame[max(int(y), 0):min(int(y + h), H), max(int(x), 0):min(int(x + w), W)]
                    if min(c.shape[:2]) >= 8:
                        crops.append(c.copy()), meta.append((f"{scn}_{tid}", f"{scn}_{cam.name}", w * h))
            del masks
            used, total = used + n_masked, total + len(crops)
            print(f"{scn}/{cam.name}: {len(crops)} crops" + (f", {n_masked} with background removed" if masked else ""))
            if crops:
                feats.append(clip_embed(crops))
                labels += [m[0] for m in meta]; cams += [m[1] for m in meta]; areas += [m[2] for m in meta]
    if masked:
        print(f"background removed on {used} of {total} crops ({used / max(total, 1):.0%}); the rest had no matching mask")
    data = {"feats": np.concatenate(feats).astype(np.float32), "labels": np.array(labels), "cams": np.array(cams),
            "areas": np.array(areas, dtype=np.float32)}
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


# ============================================================================================================================
# Re-identification: tracklets, retrieval and the rules for showing a match (numpy only)
# ============================================================================================================================
def tracklets(data):
    """One vector per (camera, vehicle): mean of its biggest crops, normalised. What multicam.py stores for every track."""
    groups = defaultdict(list)
    for i, key in enumerate(zip(data["cams"], data["labels"])):
        groups[key].append(i)
    cams, labs, vecs = [], [], []
    for (c, l), idx in groups.items():
        idx = sorted(idx, key=lambda i: -data["areas"][i])[:EMBED_CROPS]
        cams.append(c), labs.append(l), vecs.append(unit(data["feats"][idx].mean(axis=0)))
    return np.array(cams), np.array(labs), np.array(vecs, dtype=np.float32)


def camera_center(vecs, cams):
    """Subtract each camera's own mean vector (its lighting/viewpoint/colour cast), renormalise. Same as multicam.py's CAMERA_CENTER."""
    out = vecs.copy()
    for c in np.unique(cams):
        m = cams == c
        if m.sum() >= 5:
            out[m] = vecs[m] - vecs[m].mean(axis=0)
    return unit(out)


def retrieve(cams, labs, vecs, rows=False, pen=None, w=0.0):
    """Cross-camera retrieval inside each scenario. rank1/mAP: rank the tracklets of all other cameras for every query.
    rows=True also gives what the viewer does: for every other camera, its single most similar track, and whether that pick
    is mutual (looking back from it, the query is the best match in the query's own camera).
    pen: matrix of gate penalties (0 plausible .. 1 impossible). Impossible pairs are ruled out, the others lose w * penalty."""
    scen = np.array([c.split("_")[0] for c in cams])
    S = vecs @ vecs.T
    if pen is not None:
        S = S - w * pen
        S[pen >= 1.0] = EXCLUDED
    members = {c: np.where(cams == c)[0] for c in set(cams)}
    cam_list = {s: sorted(set(cams[scen == s])) for s in set(scen)}
    ap, r1, sims, correct, present, mutual = [], [], [], [], [], []
    for q in range(len(cams)):
        gal = np.where((scen == scen[q]) & (cams != cams[q]))[0]
        hit = labs[gal] == labs[q]
        if not hit.any():
            continue                                               # vehicle seen by no other camera: nothing to find
        h = hit[np.argsort(-S[q, gal])]
        r1.append(float(h[0]))
        ap.append(float((np.cumsum(h)[h] / (np.flatnonzero(h) + 1)).mean()))
        if rows:
            mine = members[cams[q]]
            for c in cam_list[scen[q]]:
                if c != cams[q]:
                    m = members[c]
                    j = m[np.argmax(S[q, m])]
                    sims.append(S[q, j]), correct.append(labs[j] == labs[q]), present.append(bool((labs[m] == labs[q]).any()))
                    mutual.append(mine[np.argmax(S[j, mine])] == q)
    return {"rank1": float(np.mean(r1)), "mAP": float(np.mean(ap)), "queries": len(r1), "sims": np.array(sims),
            "correct": np.array(correct, bool), "present": np.array(present, bool), "mutual": np.array(mutual, bool)}


def lowest_threshold(sims, ok, mask):
    """Lowest REID_MIN at which TARGET_PRECISION of the shown picks (among `mask`) are right, keeping 5% of all picks at least."""
    if not mask.any():
        return None
    for t in np.unique(np.quantile(sims[mask], np.linspace(0, 0.95, 96))):
        s = mask & (sims >= t)
        if s.sum() < 0.05 * len(sims):
            break
        if ok[s].mean() >= TARGET_PRECISION:
            return float(t)
    return None


def gate_report(r):
    """The viewer's ways of hiding wrong picks. A pick is wrong when it is another vehicle OR when the vehicle is not in that camera.
    Rules: a minimum similarity (REID_MIN), and/or the mutual-best-match test (the k=1 case of the k-reciprocal rule the winning
    AI City 2022 system used). Picks the spatio-temporal gate ruled out are never shown. Returns (RECIPROCAL, REID_MIN, reached_target)."""
    sims, ok, present, mutual = r["sims"], r["correct"], r["present"], r["mutual"]
    valid = sims > EXCLUDED / 2                                        # False where the gate left no candidate at all
    base = {False: valid, True: valid & mutual}
    print(f"    the vehicle is in {present.mean():.0%} of the (query, other camera) cases; when it is, the top pick is right {ok[present].mean():.0%} of the time")
    t_all, t_mut = lowest_threshold(sims, ok, base[False]), lowest_threshold(sims, ok, base[True])
    rules = [("show every pick", False, None), ("mutual best match only", True, None)]
    if t_all is not None:
        rules.append((f"REID_MIN {t_all:.2f}", False, t_all))
    if t_mut is not None:
        rules.append((f"mutual + REID_MIN {t_mut:.2f}", True, t_mut))
    print(f"    {'rule':30s} {'picks shown':>11s} {'right among shown':>19s} {'vehicles found':>15s}")
    scored = []
    for name, mut, t in rules:
        shown = base[mut] & (sims >= (-9 if t is None else t))
        prec = float(ok[shown].mean()) if shown.any() else 0.0
        found = float((shown & ok).sum() / present.sum())
        print(f"    {name:30s} {shown.mean():11.0%} {prec:19.0%} {found:15.0%}")
        scored.append((mut, t, float(shown.mean()), prec, found))
    good = [s for s in scored if s[3] >= TARGET_PRECISION and s[2] >= 0.05]
    pick = max(good, key=lambda s: (round(s[4], 3), s[3])) if good else max((s for s in scored if s[2] >= 0.05), key=lambda s: s[3], default=scored[0])
    return pick[0], (0.0 if pick[1] is None else pick[1]), bool(good)


# ============================================================================================================================
# Spatio-temporal gate: summaries of the labelled tracks, fitting the thresholds, checking them
# ============================================================================================================================
def read_offsets(scn):
    path = OTHERS / "cam_timestamp" / f"{scn}.txt"
    if not path.exists():
        raise SystemExit(f"Can't find {path}\nSet OTHERS at the top of this file to the folder that contains cam_timestamp/.")
    return {name: float(sec) for name, sec in (line.split() for line in path.read_text().splitlines() if line.strip())}


def gt_summaries(scenarios):
    """{(camera, vehicle): track summary} of every labelled track, with the same keys as the tracklets of the re-ID data."""
    out = {}
    for split, scn in scenarios:
        offsets = read_offsets(scn)
        for cam in cameras(split, scn):
            if cam.name not in offsets or not (cam / "calibration.txt").exists():
                print(f"    {scn}/{cam.name}: no timestamp or calibration, the gate cannot judge its tracks")
                continue
            roi = load_roi(cam / "roi.jpg") if (cam / "roi.jpg").exists() else None
            calib = Calibration(cam / "calibration.txt", (roi.shape[1], roi.shape[0]) if roi is not None else (1920, 1080))
            gt = load_mot(cam / "gt" / "gt.txt")
            rows = np.vstack([np.c_[np.full(len(r), f), r] for f, r in gt.items()])        # frame, id, x, y, w, h, conf
            lat, lon, ok = calib.pixels_to_gps(np.c_[rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5]])   # bottom centre of each box
            xy = np.c_[to_metres(lat, lon, SCENARIO_CENTER[scn])]
            t = offsets[cam.name] + (rows[:, 0] - 1) / FPS.get(cam.name, DEFAULT_FPS)
            for tid in np.unique(rows[:, 1]):
                m = (rows[:, 1] == tid) & ok
                order = np.argsort(t[m])
                out[(f"{scn}_{cam.name}", f"{scn}_{int(tid)}")] = summarize_track(t[m][order], xy[m][order])
    return out


def fit_gate(summ):
    """Thresholds from the labelled pairs: where one vehicle really was seen by two cameras, how far apart were the positions at the
    same moment, how long between the cameras, how fast, which way. Each threshold sits at the 95th-99th percentile of those true pairs."""
    by = defaultdict(list)
    for (c, l), s in summ.items():
        if s is not None:
            by[l].append((c, s))
    A, B = [], []
    for lst in by.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] != lst[j][0]:
                    A.append(lst[i][1]), B.append(lst[j][1])
    if len(A) < 30:
        print("    too few labelled cross-camera pairs to fit the gate: using the defaults")
        return dict(GATE)
    q = pair_quantities(stack_tracks(A), stack_tracks(B), outer=False)
    ov, seq = q["overlap"], ~q["overlap"]
    g = dict(GATE)
    if ov.sum() >= 20:
        g["overlap_m"] = float(np.clip(np.percentile(q["d_ov"][ov], 95), 8, 40))
    if seq.sum() >= 20:
        g["max_gap"] = float(np.clip(np.percentile(q["gap"][seq], 99), 30, 400))
        g["max_speed"] = float(np.clip(np.percentile(q["v_need"][seq], 99), 10, 50))
        far = seq & (q["d"] > g["min_dist"])
        cos = np.r_[q["cos_exit"][far & (q["sp_exit"] > g["min_speed"])], q["cos_entry"][far & (q["sp_entry"] > g["min_speed"])]]
        if len(cos) >= 20:
            g["cos_ok"] = float(np.clip(np.percentile(cos, 5), -0.5, 0.5))
            g["cos_bad"] = max(g["cos_ok"] - 0.6, -1.0)
    print(f"    fitted on {len(A)} cross-camera pairs of the same vehicle ({int(ov.sum())} at the same moment, {int(seq.sum())} one after the other):")
    print("    " + "  ".join(f"{k} {v:.2f}" for k, v in g.items()))
    return g


def pen_matrix(cams, labs, summ, gate):
    """Gate penalty of every tracklet against every other one of its scenario (computed in row blocks to keep memory small)."""
    scen = np.array([c.split("_")[0] for c in cams])
    pen = np.zeros((len(cams), len(cams)), np.float32)
    for s in set(scen):
        idx = np.where(scen == s)[0]
        st = stack_tracks([summ.get((str(cams[i]), str(labs[i]))) for i in idx])
        for a in range(0, len(idx), 256):
            sub = {k: v[a:a + 256] for k, v in st.items()}
            pen[np.ix_(idx[a:a + 256], idx)] = gate_penalty(pair_quantities(sub, st, p=gate), gate)
    return pen


def gate_stats(cams, labs, pen):
    scen = np.array([c.split("_")[0] for c in cams])
    pair = (scen[:, None] == scen[None]) & (cams[:, None] != cams[None])
    same = labs[:, None] == labs[None]
    true, false = pair & same, pair & ~same
    print(f"    same vehicle, different cameras: {(pen[true] < 1).mean():.1%} kept by the gate ({true.sum() // 2} pairs)")
    print(f"    different vehicles:              {(pen[false] >= 1).mean():.1%} ruled out ({false.sum() // 2} pairs)")


# ============================================================================================================================
# Re-identification: the head and its loss (torch)
# ============================================================================================================================
class Head(nn.Module):
    """Centre, then a residual linear layer. W = 0 at the start, so the untrained head is plain centred CLIP."""

    def __init__(self, mu, d=512):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.lin = nn.Linear(d, d, bias=False)                      # no bias: a bias is what let the first version collapse
        nn.init.zeros_(self.lin.weight)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        z = F.normalize(x - self.mu, dim=-1)
        return F.normalize(z + self.lin(self.drop(z)), dim=-1)


def cross_camera_supcon(emb, y, c):
    """Supervised contrastive loss. Positives: same vehicle, different camera. Same vehicle in the same camera is neither
    positive nor negative (it says nothing about camera-to-camera matching). Everything else is a negative."""
    same_id, same_cam = y[:, None] == y[None], c[:, None] == c[None]
    pos, ignore = same_id & ~same_cam, same_id & same_cam           # `ignore` includes the diagonal
    logp = (emb @ emb.T / TEMP).masked_fill(ignore, -1e9)
    logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    has = pos.any(dim=1)
    return -(logp.masked_fill(~pos, 0.0).sum(1)[has] / pos.sum(1)[has]).mean()


def apply_head(head, v):
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(np.asarray(v, dtype=np.float32))).numpy()


def train_head(tr, mu, monitor=None):
    rng = np.random.default_rng(SEED)
    X = torch.from_numpy(tr["feats"])
    pools = defaultdict(lambda: defaultdict(list))                  # vehicle -> camera -> crop indices
    for i, (l, c) in enumerate(zip(tr["labels"], tr["cams"])):
        pools[str(l)][str(c)].append(i)
    pools = {l: {c: np.array(v) for c, v in d.items()} for l, d in pools.items() if len(d) >= 2}
    by_scen = defaultdict(list)
    for l in pools:
        by_scen[l.split("_")[0]].append(l)
    scens = sorted(by_scen)
    weight = np.array([len(by_scen[s]) for s in scens], float)
    weight /= weight.sum()
    vcode = {l: i for i, l in enumerate(pools)}
    ccode = {str(c): i for i, c in enumerate(sorted(set(tr["cams"])))}
    print(f"    {len(pools)} vehicles seen by 2+ cameras in {len(scens)} scenarios")

    head = Head(mu)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(1, ITERS + 1):
        head.train()
        s = scens[rng.choice(len(scens), p=weight)]                 # one scenario per batch: negatives look like the real gallery
        chosen = rng.choice(by_scen[s], size=min(P, len(by_scen[s])), replace=False)
        idx, y, c = [], [], []
        for l in chosen:
            for cam in rng.choice(list(pools[l]), size=2, replace=False):
                pool = pools[l][str(cam)]
                for _ in range(K // 2):                             # a tracklet = mean of 5 random crops of this vehicle in this camera
                    idx.append(pool[rng.integers(len(pool), size=EMBED_CROPS)]), y.append(vcode[l]), c.append(ccode[str(cam)])
        v = F.normalize(X[torch.from_numpy(np.stack(idx))].mean(dim=1), dim=-1)
        loss = cross_camera_supcon(head(v), torch.tensor(y), torch.tensor(c))
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        if it % 250 == 0:
            msg = f"    iter {it:5d}  loss {loss.item():.3f}"
            if monitor:
                r = monitor(head)
                msg += f"   validation rank-1 {r['rank1']:.3f}  mAP {r['mAP']:.3f}  (monitoring only, the last iteration is kept)"
            print(msg)
    return head


def reid(mode):
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    tr, va = collect(TRAIN, "train", USE_MASKS), collect(VAL, "val", USE_MASKS)
    fit_on = TRAIN + (VAL if mode == "all" else [])
    if mode == "all":
        tr = {k: np.concatenate([tr[k], va[k]]) for k in tr}
    mu = tracklets(tr)[2].mean(axis=0)                              # mean over training TRACKS, like what the viewer feeds the head
    print(f"{len(set(tr['labels']))} vehicles, {len(tr['feats'])} crops to train on")
    print("Spatio-temporal gate:")
    gate = fit_gate(gt_summaries(fit_on))

    if mode == "all":
        head = train_head(tr, mu)
        np.savez(HEAD_OUT_ALL, mu=mu, W=head.lin.weight.detach().numpy())
        GATE_OUT_ALL.write_text(json.dumps(gate, indent=1))
        print(f"saved {HEAD_OUT_ALL} and {GATE_OUT_ALL}\nUse the CAMERA_CENTER / RECIPROCAL / REID_MIN / ST_WEIGHT printed by the plain `py finetune.py reid` run.")
        return

    VC, VL, VV = tracklets(va)
    print(f"{len(set(va['labels']))} validation vehicles -> {len(VC)} tracklets, {len(set(VC))} cameras")
    PEN = pen_matrix(VC, VL, gt_summaries(VAL), gate)
    print("How the gate treats the validation vehicles:")
    gate_stats(VC, VL, PEN)

    show = lambda name, r: print(f"  {name:36s} rank-1 {r['rank1']:.3f}   mAP {r['mAP']:.3f}")
    print("\nValidation (S02, S05): cross-camera retrieval")
    if USE_MASKS and (WORK / "reid_val.npz").exists():              # the earlier plain-crop features, only for this comparison
        pc, pl, pv = tracklets(collect(VAL, "val", False))
        show("CLIP, plain crops (before)", retrieve(pc, pl, pv))
    show("CLIP" + (", background removed" if USE_MASKS else ""), retrieve(VC, VL, VV))
    show("CLIP, minus global mean", retrieve(VC, VL, unit(VV - mu)))
    show("CLIP, minus each camera's mean", retrieve(VC, VL, camera_center(VV, VC)))

    print("\nTraining the head ...")
    head = train_head(tr, mu, monitor=lambda h: retrieve(VC, VL, apply_head(h, VV)))
    H = apply_head(head, VV)
    r_head, r_cam = retrieve(VC, VL, H, rows=True), retrieve(VC, VL, camera_center(H, VC), rows=True)
    print("\nValidation (S02, S05), after training:")
    show("head", r_head)
    show("head + each camera's mean", r_cam)
    use_cam = r_cam["mAP"] > r_head["mAP"]
    best, r_plain = (camera_center(H, VC), r_cam) if use_cam else (H, r_head)

    print(f"\nAdding the spatio-temporal gate to the best setting ({'head + camera centering' if use_cam else 'head'}):")
    gated = {w: retrieve(VC, VL, best, rows=True, pen=PEN, w=w) for w in ST_WEIGHTS}
    for w, r in gated.items():
        show(f"+ gate, soft penalty weight {w}", r)
    w_best = max(gated, key=lambda w: gated[w]["mAP"])

    print("\nWhat the viewer does, WITHOUT the gate:")
    gate_report(r_plain)
    print(f"\nWhat the viewer does, WITH the gate (weight {w_best}):")
    reciprocal, reid_min, reached = gate_report(gated[w_best])

    np.savez(HEAD_OUT, mu=mu, W=head.lin.weight.detach().numpy())
    GATE_OUT.write_text(json.dumps(gate, indent=1))
    print(f"\nsaved {HEAD_OUT} and {GATE_OUT}")
    print(f"Set in multicam.py:   USE_MASKS = {USE_MASKS}   CAMERA_CENTER = {use_cam}   RECIPROCAL = {reciprocal}   "
          f"REID_MIN = {reid_min:.2f}   ST_GATE = True   ST_WEIGHT = {w_best}")
    if not reached:
        print(f"(no rule reaches {TARGET_PRECISION:.0%} right: this is the best precision available, read the tables)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"det": lambda: detector(*sys.argv[2:3]), "reid": lambda: reid("val"), "reid-all": lambda: reid("all")}.get(cmd, lambda: sys.exit(__doc__))()
