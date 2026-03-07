import cv2
import base64
import json
import os
import sys
import numpy as np
from typing import List, Dict, Set, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

load_dotenv()

# ============================================================
# Gemini schemas
# ============================================================

class GeminiBBox(BaseModel):
    ymin: int = Field(description="Y min (0-1000)")
    xmin: int = Field(description="X min (0-1000)")
    ymax: int = Field(description="Y max (0-1000)")
    xmax: int = Field(description="X max (0-1000)")

class DetectedPerson(BaseModel):
    label: str = Field(description="Description, e.g. Person 1 (Aggressor)")
    box: GeminiBBox = Field(description="Bounding box [ymin, xmin, ymax, xmax] scaled 0-1000")

class VisionAnalysis(BaseModel):
    alert: bool = Field(description="Is there a physical altercation or someone on the ground/falling?")
    reasoning: str = Field(description="Reasoning behind the alert.")
    detected_people: List[DetectedPerson] = Field(description="People involved in the incident.")


# ============================================================
# Kalman + RTS smoother — bottom-center anchor
# State: [xb, yb, w, h, vx, vy, vw, vh]
# ============================================================

def make_kalman(dt=1.0):
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8)
    kf.F[0, 4] = dt
    kf.F[1, 5] = dt
    kf.F[2, 6] = dt
    kf.F[3, 7] = dt
    kf.H = np.zeros((4, 8))
    kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = kf.H[3, 3] = 1.0
    kf.R *= 25.0       # high measurement noise = trust prediction more = smoother
    kf.P[4:, 4:] *= 100
    kf.Q[4:, 4:] *= 0.02
    kf.Q[:4, :4] *= 0.5
    return kf


def xyxy_to_anchor(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2)/2.0, float(y2), float(x2-x1), float(y2-y1)])


def anchor_to_xyxy(state):
    xb, yb, w, h = state[0], state[1], max(10, state[2]), max(10, state[3])
    return [int(round(xb - w/2)), int(round(yb - h)), int(round(xb + w/2)), int(round(yb))]


def rts_smooth_track(measurements: Dict[int, np.ndarray], all_frames: List[int]):
    """
    Forward Kalman pass then backward RTS smoothing pass.
    measurements: {frame_idx: np.array([xb, yb, w, h])}
    all_frames: sorted list of all frame indices this track spans
    Returns: {frame_idx: [x1, y1, x2, y2]} for every frame in range
    """
    if not measurements:
        return {}

    first = min(measurements.keys())
    last = max(measurements.keys())
    frames_in_range = [f for f in all_frames if first <= f <= last]
    if not frames_in_range:
        return {}

    kf = make_kalman()
    z0 = measurements[min(measurements.keys())]
    kf.x[:4] = z0.reshape(4, 1)

    # Forward pass: collect means, covariances, predictions
    xs, Ps = [], []
    Fs, Qs = [], []

    for fi in frames_in_range:
        kf.predict()
        if fi in measurements:
            kf.update(measurements[fi].reshape(4, 1))
        xs.append(kf.x.copy())
        Ps.append(kf.P.copy())
        Fs.append(kf.F.copy())
        Qs.append(kf.Q.copy())

    # Backward RTS pass
    n = len(xs)
    xs_s = [None] * n
    xs_s[-1] = xs[-1]

    for k in range(n - 2, -1, -1):
        Pp = Fs[k] @ Ps[k] @ Fs[k].T + Qs[k]
        try:
            K = Ps[k] @ Fs[k].T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            K = Ps[k] @ Fs[k].T @ np.linalg.pinv(Pp)
        xs_s[k] = xs[k] + K @ (xs_s[k + 1] - Fs[k] @ xs[k])

    result = {}
    for i, fi in enumerate(frames_in_range):
        state = xs_s[i].flatten()
        result[fi] = anchor_to_xyxy(state)

    return result


# ============================================================
# Drawing
# ============================================================

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash=10):
    edges = [
        (pt1, (pt2[0], pt1[1])), ((pt2[0], pt1[1]), pt2),
        (pt2, (pt1[0], pt2[1])), ((pt1[0], pt2[1]), pt1),
    ]
    for (sx, sy), (ex, ey) in edges:
        length = ((ex-sx)**2 + (ey-sy)**2) ** 0.5
        if length < 1:
            continue
        dx, dy = (ex-sx)/length, (ey-sy)/length
        d, on = 0.0, True
        while d < length:
            seg = min(dash, length - d)
            if on:
                cv2.line(img, (int(sx+dx*d), int(sy+dy*d)),
                         (int(sx+dx*(d+seg)), int(sy+dy*(d+seg))), color, thickness)
            d += seg
            on = not on


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(aa + ab - inter + 1e-5)


# ============================================================
# Config
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found.")
    sys.exit(1)

client = genai.Client(api_key=api_key)
gemini_model = "gemini-2.5-flash"
video_path = "4.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: cannot open {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

PROCESS_FPS = 30
SKIP = max(1, round(fps / PROCESS_FPS))

print(f"Video: {video_path} — {total_frames} frames @ {fps:.1f} fps ({W}x{H})")
print(f"Processing every {SKIP} frames (~{fps/SKIP:.0f} fps effective)")

yolo = YOLO("yolo11n.pt")
inference_interval = int((fps / SKIP) * 1.5)  # ~every 1.5 seconds in processed frames


# ============================================================
# PASS 1: YOLO + ByteTrack on sampled frames + Gemini
# No Re-ID — trust ByteTrack native IDs only
# ============================================================

print("\n========== PASS 1: Detection + Tracking ==========\n")

# raw_tracks[track_id][frame_idx] = [x1, y1, x2, y2]
raw_tracks: Dict[int, Dict[int, np.ndarray]] = {}
all_processed_frames: List[int] = []
conflict_ids: Set[int] = set()
altercation_seconds: Set[float] = set()

cap = cv2.VideoCapture(video_path)
frame_idx = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SKIP != 0:
        frame_idx += 1
        continue

    t_sec = frame_idx / fps
    all_processed_frames.append(frame_idx)

    results = yolo.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[0], verbose=False, conf=0.35)

    frame_detections = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for bbox, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = bbox
            z = xyxy_to_anchor([x1, y1, x2, y2])
            if tid not in raw_tracks:
                raw_tracks[tid] = {}
            raw_tracks[tid][frame_idx] = z
            frame_detections.append((tid, [int(x1), int(y1), int(x2), int(y2)]))

    # Gemini inference
    if processed_count % inference_interval == 0:
        _, buf = cv2.imencode('.jpg', frame)
        image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg')
        prompt = (
            "Analyze this surveillance frame. Is there a physical altercation, fight, or someone "
            "on the ground / falling? Identify involved people with bounding boxes "
            "[ymin, xmin, ymax, xmax] normalized 0-1000."
        )
        try:
            resp = client.models.generate_content(
                model=gemini_model, contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisionAnalysis, temperature=0.0,
                ),
            )
            analysis = json.loads(resp.text)
            if analysis.get("alert"):
                altercation_seconds.add(t_sec)
                for person in analysis.get("detected_people", []):
                    b = person["box"]
                    gbox = [int((b["xmin"]/1000)*W), int((b["ymin"]/1000)*H),
                            int((b["xmax"]/1000)*W), int((b["ymax"]/1000)*H)]
                    best_iou, best_tid = 0.0, None
                    for tid, tbox in frame_detections:
                        v = iou_xyxy(gbox, tbox)
                        if v > best_iou:
                            best_iou, best_tid = v, tid
                    if best_tid is not None and best_iou > 0.15:
                        conflict_ids.add(best_tid)
                print(f"[{t_sec:>5.1f}s] ALERT | Conflict IDs: {sorted(conflict_ids)}")
            else:
                print(f"[{t_sec:>5.1f}s] Clear.")
        except Exception as e:
            print(f"[{t_sec:>5.1f}s] Gemini error: {e}")

    processed_count += 1
    frame_idx += 1
    if processed_count % 30 == 0:
        print(f"  Processed {processed_count} frames ({t_sec:.1f}s)")

cap.release()

print(f"\nPass 1 complete: {len(raw_tracks)} tracks, conflict IDs: {sorted(conflict_ids)}")


# ============================================================
# PASS 2: RTS smooth every track + interpolate to 60fps
# ============================================================

print("\n========== PASS 2: RTS Smoothing + Interpolation ==========\n")

# smoothed_output[frame_idx] = [(track_id, [x1,y1,x2,y2]), ...]
smoothed_output: Dict[int, List[Tuple[int, List[int]]]] = {}

for tid, measurements in raw_tracks.items():
    smoothed = rts_smooth_track(measurements, all_processed_frames)
    if not smoothed:
        continue

    sorted_frames = sorted(smoothed.keys())

    # Interpolate to all 60fps frames between first and last
    for i in range(len(sorted_frames) - 1):
        f_start = sorted_frames[i]
        f_end = sorted_frames[i + 1]
        box_start = smoothed[f_start]
        box_end = smoothed[f_end]
        gap = f_end - f_start

        for f in range(f_start, f_end + 1):
            t = (f - f_start) / gap if gap > 0 else 0
            box = [int(round(box_start[k] + t * (box_end[k] - box_start[k]))) for k in range(4)]
            if f not in smoothed_output:
                smoothed_output[f] = []
            smoothed_output[f].append((tid, box))

    # Add the last frame if not covered
    last_f = sorted_frames[-1]
    if last_f not in smoothed_output:
        smoothed_output[last_f] = []
    smoothed_output[last_f].append((tid, smoothed[last_f]))

print(f"Smoothed {len(raw_tracks)} tracks, interpolated to {len(smoothed_output)} frames")


# ============================================================
# PASS 3: Annotate
# ============================================================

print("\n========== PASS 3: Annotating ==========\n")

output_path = "new67anno.avi"
codecs = [('MJPG', 'new67anno.avi'), ('avc1', 'new67anno.mp4'), ('mp4v', 'new67anno.mp4')]
out = None
for codec, path in codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if out.isOpened():
        output_path = path
        print(f"Codec {codec} -> {output_path}")
        break

if out is None or not out.isOpened():
    print("Error: no codec.")
    sys.exit(1)

GREEN = (0, 200, 0)
RED = (0, 0, 255)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_sec = frame_idx / fps
    tracks = smoothed_output.get(frame_idx, [])

    # Clamp
    clamped = []
    for tid, box in tracks:
        x1 = max(0, min(box[0], W-1))
        y1 = max(0, min(box[1], H-1))
        x2 = max(x1+1, min(box[2], W))
        y2 = max(y1+1, min(box[3], H))
        clamped.append((tid, [x1, y1, x2, y2]))
    tracks = clamped

    # NMS per track ID (shouldn't be needed but safety)
    best_per_id: Dict[int, Tuple[int, List[int]]] = {}
    for tid, box in tracks:
        area = (box[2]-box[0]) * (box[3]-box[1])
        if tid not in best_per_id:
            best_per_id[tid] = (tid, box)
        else:
            old = best_per_id[tid][1]
            if area > (old[2]-old[0]) * (old[3]-old[1]):
                best_per_id[tid] = (tid, box)
    tracks = list(best_per_id.values())

    # Z-order: dashed for occluded
    occluded: Set[int] = set()
    for i, (a_id, a_box) in enumerate(tracks):
        for j, (b_id, b_box) in enumerate(tracks):
            if i >= j:
                continue
            if iou_xyxy(a_box, b_box) > 0.15:
                a_area = (a_box[2]-a_box[0]) * (a_box[3]-a_box[1])
                b_area = (b_box[2]-b_box[0]) * (b_box[3]-b_box[1])
                occluded.add(a_id if a_area < b_area else b_id)

    for tid, box in tracks:
        x1, y1, x2, y2 = box
        is_conflict = tid in conflict_ids
        color = RED if is_conflict else GREEN
        thick = 3 if is_conflict else 2

        if tid in occluded:
            draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thick)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"ID {tid}"
        if is_conflict:
            label += " [CONFLICT]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, max(0, y1-th-6)), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, max(th+2, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if any(tid in conflict_ids for tid, _ in tracks):
        cv2.putText(frame, "ALERT: ALTERCATION DETECTED", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    cv2.putText(frame, f"Time: {t_sec:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 120 == 0:
        print(f"  Annotating: {frame_idx}/{total_frames} ({t_sec:.1f}s)")

cap.release()
out.release()
print(f"\nDone -> {output_path}")

formatted = sorted([round(s, 1) for s in altercation_seconds])
print(f"Altercation timestamps: {formatted}")
with open("altercation_report.txt", "w") as f:
    f.write(f"Conflict IDs: {sorted(conflict_ids)}\n")
    f.write(f"Altercation timestamps: {formatted}\n")
