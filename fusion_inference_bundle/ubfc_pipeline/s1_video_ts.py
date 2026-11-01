"""Stage 1: stream each video once and keep only small time series (.npz, ~100 KB each).
No frames are ever written to disk; memory use is one frame at a time.

Per video:
  * every `stride`-th frame -> MediaPipe FaceLandmarker: 52 blendshapes + head pose (3 Euler angles)
  * every frame (unless --no-rppg) -> mean RGB over forehead+cheek ROIs (for POS rPPG in stage 2)

Setup (once): pip install mediapipe opencv-python
  download the official 'face_landmarker.task' model (~4 MB) from the MediaPipe model page and pass --model.
Resumable: existing .npz files are skipped. Test first:  --subjects 1 --limit-frames 600
Rough cost: decoding ~1M+ frames of 1024px video on an M2 is hours -> run overnight, or use --no-rppg
(then only landmark frames are decoded: much faster, but you lose rPPG HR / the video-vs-BVP sync check).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import index_files


def euler_deg(M):
    R = np.array(M)[:3, :3]
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    sy = np.hypot(R[0, 0], R[1, 0])
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])


def roi_boxes(lm, aspect):
    """Normalised (x0,x1,y0,y1) boxes: forehead + both cheeks, from FaceMesh landmark indices."""
    fw = abs(lm[454, 0] - lm[234, 0])
    d = lm[9, 1] - lm[10, 1]
    boxes = [(lm[66, 0], lm[296, 0], lm[10, 1] + 0.2 * d, lm[9, 1] - 0.2 * d)]
    hx = 0.06 * fw
    hy = hx * aspect
    for i in (50, 280):
        boxes.append((lm[i, 0] - hx, lm[i, 0] + hx, lm[i, 1] - hy, lm[i, 1] + hy))
    return boxes


def roi_mean(img, boxes):
    h, w = img.shape[:2]
    vals = []
    for x0, x1, y0, y1 in boxes:
        a, b, c, d = int(max(0, x0 * w)), int(min(w, x1 * w)), int(max(0, y0 * h)), int(min(h, y1 * h))
        if b - a >= 2 and d - c >= 2:
            vals.append(img[c:d, a:b].reshape(-1, 3).mean(0))
    return np.mean(vals, 0) if vals else np.full(3, np.nan)


def process(vpath, opath, model, stride, do_rppg, limit):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    lmk = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_lm, bs_l, pose_l, rgb_l, names = [], [], [], [], None
    boxes, i = None, 0
    while True:
        need_lm = i % stride == 0
        if limit and i >= limit:
            break
        if not do_rppg and not need_lm:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        sc = 480.0 / max(h, w)
        small = cv2.cvtColor(cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        if need_lm:
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), int(i / fps * 1000))
            t_lm.append(i / fps)
            if res.face_landmarks:
                lm = np.array([[p.x, p.y] for p in res.face_landmarks[0]])
                boxes = roi_boxes(lm, small.shape[1] / small.shape[0])
                cats = res.face_blendshapes[0]
                if names is None:
                    names = [c.category_name for c in cats]
                bs_l.append([c.score for c in cats])
                pose_l.append(euler_deg(res.facial_transformation_matrixes[0]))
            else:
                boxes = None
                bs_l.append(None)
                pose_l.append(None)
        if do_rppg:
            rgb_l.append(roi_mean(small, boxes) if boxes is not None else np.full(3, np.nan))
        i += 1
    cap.release()
    lmk.close()
    nb = len(names) if names else 52
    bs = np.array([b if b is not None else [np.nan] * nb for b in bs_l], dtype=np.float32)
    pose = np.array([p if p is not None else [np.nan] * 3 for p in pose_l], dtype=np.float32)
    np.savez_compressed(opath, fps=fps, n_frames=i, t_lm=np.array(t_lm), bs=bs, pose=pose,
                        bs_names=np.array(names or [], dtype=str),
                        rgb=np.array(rgb_l, dtype=np.float32) if do_rppg else np.zeros((0, 3), np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="out/ts")
    ap.add_argument("--model", required=True, help="path to face_landmarker.task")
    ap.add_argument("--stride", type=int, default=3, help="landmark every N frames (35fps/3 = ~12 Hz)")
    ap.add_argument("--no-rppg", action="store_true")
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--limit-frames", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for (sid, task), f in sorted(index_files(a.root).items()):
        if "vid" not in f or (a.subjects and sid not in a.subjects):
            continue
        o = Path(a.out) / f"s{sid}_{task}.npz"
        if o.exists():
            continue
        t = time.time()
        process(f["vid"], o, a.model, a.stride, not a.no_rppg, a.limit_frames)
        print(f"s{sid} {task} done in {time.time() - t:.0f}s", flush=True)
