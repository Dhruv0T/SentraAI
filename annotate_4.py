import cv2
import base64
import json
import os
import sys
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ============================================================
# Pydantic schemas for Gemini structured output
# ============================================================

class BoundingBox(BaseModel):
    ymin: int = Field(description="Y min (0-1000)")
    xmin: int = Field(description="X min (0-1000)")
    ymax: int = Field(description="Y max (0-1000)")
    xmax: int = Field(description="X max (0-1000)")

class DetectedPerson(BaseModel):
    label: str = Field(description="Description, e.g. Person 1 (Aggressor)")
    box: BoundingBox = Field(description="Bounding box [ymin, xmin, ymax, xmax] scaled 0-1000")

class VisionAnalysis(BaseModel):
    alert: bool = Field(description="Is there a physical altercation or someone on the ground/falling?")
    reasoning: str = Field(description="Reasoning behind the alert.")
    detected_people: List[DetectedPerson] = Field(description="People involved in the incident.")


# ============================================================
# Re-ID Gallery: visual fingerprints for persistent identity
# ============================================================

class IDGallery:
    def __init__(self, match_threshold: float = 0.70):
        self.gallery: Dict[int, List[np.ndarray]] = {}
        self.id_map: Dict[int, int] = {}
        self.match_threshold = match_threshold
        self.max_features = 10

    def extract_feature(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        mid = crop_resized.shape[0] // 2
        upper_hsv = cv2.cvtColor(crop_resized[:mid], cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(crop_resized[mid:], cv2.COLOR_BGR2HSV)
        hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        hist_lower = cv2.calcHist([lower_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()

        feature = np.concatenate([hist_h, hist_s, hist_v, hist_upper, hist_lower])
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        return feature

    def match_or_register(self, track_id: int, feature: np.ndarray) -> int:
        if track_id in self.id_map:
            canonical = self.id_map[track_id]
            if len(self.gallery[canonical]) < self.max_features:
                self.gallery[canonical].append(feature)
            return canonical

        best_score, best_id = 0.0, None
        for gid, features in self.gallery.items():
            avg = np.mean(features, axis=0)
            score = float(np.dot(feature, avg))
            if score > best_score:
                best_score, best_id = score, gid

        if best_score >= self.match_threshold and best_id is not None:
            self.id_map[track_id] = best_id
            if len(self.gallery[best_id]) < self.max_features:
                self.gallery[best_id].append(feature)
            print(f"  [Re-ID Gallery] ByteTrack #{track_id} -> canonical ID {best_id} (similarity: {best_score:.3f})")
            return best_id

        self.id_map[track_id] = track_id
        self.gallery[track_id] = [feature]
        print(f"  [Re-ID Gallery] New person registered: ID {track_id}")
        return track_id

    def dump_gallery(self):
        print(f"\n{'='*50}")
        print(f"  ID GALLERY — {len(self.gallery)} unique people")
        print(f"{'='*50}")
        for gid in sorted(self.gallery.keys()):
            n = len(self.gallery[gid])
            mapped_from = [k for k, v in self.id_map.items() if v == gid]
            print(f"  ID {gid:>3d}: {n} feature vectors | ByteTrack IDs mapped here: {mapped_from}")
        print(f"{'='*50}\n")


# ============================================================
# Helpers
# ============================================================

def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(aa + ab - inter + 1e-5)


def nms_by_canonical_id(tracks: List[Tuple[int, List[int]]]) -> List[Tuple[int, List[int]]]:
    """Keep only the largest box per canonical ID."""
    best: Dict[int, Tuple[int, List[int]]] = {}
    for cid, box in tracks:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if cid not in best or area > (best[cid][1][2] - best[cid][1][0]) * (best[cid][1][3] - best[cid][1][1]):
            best[cid] = (cid, box)
    return list(best.values())


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

print(f"Video: {video_path} — {total_frames} frames @ {fps:.1f} fps ({W}x{H})")

yolo = YOLO("yolo11n.pt")
gallery = IDGallery(match_threshold=0.70)

inference_interval_frames = int(fps * 1.5)

# ============================================================
# PASS 1: YOLO + ByteTrack tracking  +  Gemini conflict detect
# ============================================================

print("\n========== PASS 1: Tracking + Conflict Detection ==========\n")

all_frame_tracks: Dict[int, List[Tuple[int, List[int]]]] = {}
conflict_ids: Set[int] = set()
altercation_seconds: Set[float] = set()

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_sec = frame_idx / fps

    # --- YOLO + ByteTrack ---
    results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False, conf=0.3)

    frame_tracks: List[Tuple[int, List[int]]] = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for bbox, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = bbox
            feat = gallery.extract_feature(frame, bbox)
            if feat is not None:
                cid = gallery.match_or_register(tid, feat)
            else:
                cid = gallery.id_map.get(tid, tid)
            frame_tracks.append((cid, [int(x1), int(y1), int(x2), int(y2)]))

    frame_tracks = nms_by_canonical_id(frame_tracks)
    all_frame_tracks[frame_idx] = frame_tracks

    # --- Gemini inference at intervals ---
    if frame_idx % inference_interval_frames == 0:
        _, buf = cv2.imencode('.jpg', frame)
        image_part = types.Part.from_bytes(
            data=buf.tobytes(),
            mime_type='image/jpeg',
        )

        prompt = (
            "Analyze this surveillance frame. Is there a physical altercation, fight, or someone "
            "on the ground / falling? If yes, identify the people involved and output their bounding "
            "boxes [ymin, xmin, ymax, xmax] normalized 0-1000. Alert should be true if ANY altercation."
        )

        try:
            resp = client.models.generate_content(
                model=gemini_model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisionAnalysis,
                    temperature=0.0,
                ),
            )
            analysis = json.loads(resp.text)

            if analysis.get("alert"):
                altercation_seconds.add(t_sec)
                people = analysis.get("detected_people", [])

                for person in people:
                    b = person["box"]
                    gx1 = int((b["xmin"] / 1000.0) * W)
                    gy1 = int((b["ymin"] / 1000.0) * H)
                    gx2 = int((b["xmax"] / 1000.0) * W)
                    gy2 = int((b["ymax"] / 1000.0) * H)
                    gemini_box = [gx1, gy1, gx2, gy2]

                    best_iou, best_cid = 0.0, None
                    for cid, tbox in frame_tracks:
                        v = iou_xyxy(gemini_box, tbox)
                        if v > best_iou:
                            best_iou, best_cid = v, cid

                    if best_cid is not None and best_iou > 0.15:
                        conflict_ids.add(best_cid)

                print(f"[{t_sec:>5.1f}s] ALERT — {len(people)} people in conflict | "
                      f"Conflict IDs so far: {sorted(conflict_ids)}")
            else:
                print(f"[{t_sec:>5.1f}s] No altercation.")

        except Exception as e:
            print(f"[{t_sec:>5.1f}s] Gemini error: {e}")

    frame_idx += 1
    if frame_idx % 60 == 0:
        print(f"  Pass 1: {frame_idx}/{total_frames} ({t_sec:.1f}s)")

cap.release()

print(f"\nPass 1 complete.")
print(f"Conflict IDs (red boxes): {sorted(conflict_ids)}")
gallery.dump_gallery()


# ============================================================
# PASS 2: Annotate video with green / red boxes
# ============================================================

print("========== PASS 2: Annotating Video ==========\n")

output_path = "new67anno.avi"
codecs = [('MJPG', 'new67anno.avi'), ('avc1', 'new67anno.mp4'), ('mp4v', 'new67anno.mp4')]
out = None
for codec, path in codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if out.isOpened():
        output_path = path
        print(f"Using codec {codec} -> {output_path}")
        break
    print(f"  {codec} failed")

if out is None or not out.isOpened():
    print("Error: no working codec.")
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
    tracks = all_frame_tracks.get(frame_idx, [])

    for cid, box in tracks:
        x1, y1, x2, y2 = box
        is_conflict = cid in conflict_ids
        color = RED if is_conflict else GREEN
        thickness = 3 if is_conflict else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {cid}"
        if is_conflict:
            label += " [CONFLICT]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if any(cid in conflict_ids for cid, _ in tracks):
        cv2.putText(frame, "ALERT: ALTERCATION DETECTED", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    cv2.putText(frame, f"Time: {t_sec:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 120 == 0:
        print(f"  Pass 2: {frame_idx}/{total_frames} ({t_sec:.1f}s)")

cap.release()
out.release()
print(f"\nFinished -> {output_path}")

formatted = sorted([round(s, 1) for s in altercation_seconds])
print(f"Altercation timestamps: {formatted}")
with open("altercation_report.txt", "w") as f:
    f.write(f"Conflict IDs: {sorted(conflict_ids)}\n")
    f.write(f"Altercation timestamps: {formatted}\n")
