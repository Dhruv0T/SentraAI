import cv2
import argparse
import json
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

load_dotenv()

# ============================================================
# Personas — each defines what Gemini looks for
# ============================================================

PERSONAS = {
    "fight": {
        "name": "Fight Detection",
        "inference_interval_sec": 1.5,
        "prompt": (
            "Analyze this surveillance frame. Is there a physical altercation, fight, "
            "or someone on the ground / falling? Identify people involved with bounding "
            "boxes [ymin, xmin, ymax, xmax] normalized 0-1000."
        ),
        "alert_label": "ALTERCATION DETECTED",
        "tag": "CONFLICT",
        "schema_alert_desc": "Is there a physical altercation, fight, or someone on the ground/falling?",
        "schema_person_desc": "Person involved in the altercation",
    },
    "security": {
        "name": "Store Security",
        "inference_interval_sec": 0.5,  # denser sampling to catch fast grab-and-run
        "prompt": (
            "You are an expert retail loss-prevention officer reviewing store CCTV. "
            "CRITICAL: Flag anyone RUNNING at high speed — running in a store is a top-priority "
            "theft indicator (grab-and-run, snatch theft). Also flag: grab-and-run, snatch theft, "
            "someone running with merchandise, rapid exit with unpaid items, concealing merchandise, "
            "furtive movements, hiding items in bags/clothing, loitering near high-value items, "
            "or moving quickly toward exits with goods. When in doubt about someone running, FLAG THEM. "
            "Identify suspicious people with bounding boxes [ymin, xmin, ymax, xmax] normalized 0-1000."
        ),
        "alert_label": "SUSPICIOUS ACTIVITY",
        "tag": "SUSPECT",
        "schema_alert_desc": "Is there shoplifting, theft, concealment of merchandise, or suspicious behavior?",
        "schema_person_desc": "Person involved in suspicious activity",
    },
    "general": {
        "name": "General Surveillance",
        "inference_interval_sec": 1.5,
        "prompt": (
            "You are an AI security monitor analyzing CCTV footage. Flag any dangerous, "
            "suspicious, or unusual activity: fights, theft, trespassing, vandalism, loitering, "
            "or any security concern. Identify involved people with bounding boxes "
            "[ymin, xmin, ymax, xmax] normalized 0-1000."
        ),
        "alert_label": "SECURITY ALERT",
        "tag": "FLAGGED",
        "schema_alert_desc": "Is there any dangerous, suspicious, or unusual activity?",
        "schema_person_desc": "Person involved in the flagged activity",
    },
}

# ============================================================
# Gemini schemas
# ============================================================

class GeminiBBox(BaseModel):
    ymin: int = Field(description="Y min (0-1000)")
    xmin: int = Field(description="X min (0-1000)")
    ymax: int = Field(description="Y max (0-1000)")
    xmax: int = Field(description="X max (0-1000)")

class DetectedPerson(BaseModel):
    label: str = Field(description="Description of the person")
    box: GeminiBBox = Field(description="Bounding box [ymin, xmin, ymax, xmax] scaled 0-1000")

class VisionAnalysis(BaseModel):
    alert: bool = Field(description="Is there suspicious or dangerous activity in this frame?")
    reasoning: str = Field(description="Reasoning behind the alert decision.")
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
    kf.R *= 25.0
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

    xs, Ps, Fs, Qs = [], [], [], []
    for fi in frames_in_range:
        kf.predict()
        if fi in measurements:
            kf.update(measurements[fi].reshape(4, 1))
        xs.append(kf.x.copy())
        Ps.append(kf.P.copy())
        Fs.append(kf.F.copy())
        Qs.append(kf.Q.copy())

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
# Custom persona: strengthen user role + description
# ============================================================

def strengthen_prompt(role: str, description: str) -> str:
    """
    Take user-provided role and description (which may be imperfect) and produce
    a stronger, more robust prompt that expands related actions and edge cases.
    """
    role = role.strip()
    description = description.strip()
    d_lower = description.lower()

    # Expand based on common keywords so we catch related behaviors
    expansions = []
    if any(w in d_lower for w in ("theft", "steal", "shoplift", "grab", "snatch")):
        expansions.append(
            "grab-and-run, snatch theft, running with merchandise, rapid exit with items, "
            "concealing merchandise, hiding items in bags or clothing, furtive movements. "
            "CRITICAL: Flag anyone RUNNING at high speed — running in a store/venue is a "
            "top-priority theft indicator. When in doubt about someone running, FLAG THEM."
        )
    if any(w in d_lower for w in ("fight", "altercation", "attack", "assault", "violence")):
        expansions.append(
            "physical altercations, fights, someone on the ground or falling, aggressive "
            "movements, pushing, hitting. Flag anyone involved in or fleeing from violence."
        )
    if any(w in d_lower for w in ("vandal", "damage", "trespass", "loiter")):
        expansions.append(
            "vandalism, property damage, trespassing, loitering, tampering with equipment."
        )
    if any(w in d_lower for w in ("suspicious", "unusual", "concerning")):
        expansions.append(
            "any behavior that looks suspicious, furtive, or out of the ordinary. "
            "When in doubt, FLAG the person."
        )
    if any(w in d_lower for w in ("cheat", "cheater", "cheating", "exam", "proctor", "academic")):
        expansions.append(
            "looking at another person's screen or paper, using a phone or device, "
            "looking at hidden notes or materials, copying, turning to look at neighbors, "
            "passing notes, furtive glances, hands under desk, covering their work. "
            "CRITICAL: Flag anyone whose eyes or head turn toward another test-taker or "
            "who shows furtive behavior. When in doubt, FLAG them."
        )

    if not expansions:
        expansions.append(
            "any behavior that matches or is related to what was described. "
            "When in doubt about whether someone matches, FLAG THEM."
        )

    expand_text = " ".join(expansions)
    return (
        f"You are {role} reviewing CCTV footage. "
        f"Your priority: identify people engaging in {description}. "
        f"ALSO look for related behaviors: {expand_text} "
        f"Be thorough — user descriptions may be incomplete; catch the intent. "
        f"Identify all relevant people with bounding boxes [ymin, xmin, ymax, xmax] normalized 0-1000."
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Vision AI — Persona-based surveillance annotator")
    parser.add_argument("--video", "-v", required=True, help="Input video file")
    parser.add_argument("--output", "-o", default=None, help="Output filename (default: <video>_annotated.mp4)")
    parser.add_argument("--persona", "-p", default="fight",
                        choices=list(PERSONAS.keys()),
                        help=f"Built-in persona: {', '.join(PERSONAS.keys())} (default: fight)")
    parser.add_argument("--role", "-r", default=None,
                        help="Custom role (e.g. 'a store security guard'). Use with --description.")
    parser.add_argument("--description", "-d", default=None,
                        help="Custom description of what to look for (e.g. 'shoplifting'). Use with --role.")
    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit")
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Print before/after each Gemini call to debug stalls")
    parser.add_argument("--fast", "-f", action="store_true",
                        help="Fast mode: fewer Gemini calls (2s interval), smaller images. Use for live demos.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_personas:
        print("Available personas:")
        for key, p in PERSONAS.items():
            print(f"  {key:12s} — {p['name']}")
        sys.exit(0)

    # Custom role + description override built-in persona; strengthen the prompt
    if args.role and args.description:
        persona = {
            "name": f"Custom: {args.role[:50]}{'...' if len(args.role) > 50 else ''}",
            "inference_interval_sec": 0.5,  # denser for custom (user may describe fast events)
            "prompt": strengthen_prompt(args.role, args.description),
            "alert_label": "ALERT",
            "tag": "FLAGGED",
        }
        print(f"Using custom persona (strengthened from your role + description)")
    else:
        if args.role or args.description:
            print("Warning: --role and --description must both be provided; using built-in persona.")
        persona = PERSONAS[args.persona]
    video_path = args.video
    stem = Path(video_path).stem
    output_name = args.output or f"{stem}_annotated"

    mode = "Fast" if getattr(args, 'fast', False) else "Deep analysis"
    print(f"\n{'='*60}")
    print(f"  Mode    : {mode}")
    print(f"  Persona : {persona['name']}")
    print(f"  Video   : {video_path}")
    print(f"  Output  : {output_name}.mp4")
    print(f"{'='*60}\n")

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    # Load API keys (use multiple for parallel inference)
    key_names = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")
    api_keys = []
    found = []
    for name in key_names:
        k = os.environ.get(name, "").strip()
        if k:
            api_keys.append(k)
            found.append(name)
    if not api_keys:
        print("Error: No GEMINI_API_KEY found in env.")
        sys.exit(1)
    clients = [genai.Client(api_key=k) for k in api_keys]
    gemini_model = "gemini-2.5-flash"
    print(f"Using {len(clients)} API key(s): {', '.join(found)}")

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

    infer_sec = 2.0 if getattr(args, 'fast', False) else persona.get("inference_interval_sec", 1.5)
    gemini_max_width = 640 if getattr(args, 'fast', False) else None  # smaller image = faster API
    inference_interval = max(1, int((fps / SKIP) * infer_sec))

    print(f"Video: {total_frames} frames @ {fps:.1f} fps ({W}x{H})")
    print(f"Processing every {SKIP} frames (~{fps/SKIP:.0f} fps), Gemini every {infer_sec}s" +
          (f", {gemini_max_width}px" if gemini_max_width else ""))

    yolo = YOLO("yolo11n.pt")

    # --------------------------------------------------------
    # PASS 1a: YOLO + ByteTrack (collect frames for Gemini)
    # --------------------------------------------------------

    print("\n========== PASS 1a: Detection + Tracking ==========\n")

    raw_tracks: Dict[int, Dict[int, np.ndarray]] = {}
    all_processed_frames: List[int] = []
    inference_tasks: List[Tuple[float, bytes, List[Tuple[int, List[int]]]]] = []

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

        if processed_count % inference_interval == 0:
            img_for_gemini = frame
            if gemini_max_width is not None and frame.shape[1] > gemini_max_width:
                scale = gemini_max_width / frame.shape[1]
                new_w = gemini_max_width
                new_h = int(frame.shape[0] * scale)
                img_for_gemini = cv2.resize(frame, (new_w, new_h))
            _, buf = cv2.imencode('.jpg', img_for_gemini)
            inference_tasks.append((t_sec, buf.tobytes(), frame_detections))

        processed_count += 1
        frame_idx += 1
        if processed_count % 30 == 0:
            print(f"  Processed {processed_count} frames ({t_sec:.1f}s)")

    cap.release()
    print(f"  Collected {len(inference_tasks)} frames for Gemini")

    # --------------------------------------------------------
    # PASS 1b: Parallel Gemini inference
    # --------------------------------------------------------

    print("\n========== PASS 1b: Gemini (parallel) ==========\n")

    def run_gemini(args_tuple):
        client_idx, t_sec, img_bytes, frame_det, prompt = args_tuple
        c = clients[client_idx]
        try:
            image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
            resp = c.models.generate_content(
                model=gemini_model, contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VisionAnalysis, temperature=0.0,
                ),
            )
            return (t_sec, frame_det, json.loads(resp.text))
        except Exception as e:
            return (t_sec, frame_det, {"alert": False, "error": str(e)})

    flagged_ids: Set[int] = set()
    alert_seconds: Set[float] = set()
    prompt = persona["prompt"]

    task_args = [
        (i % len(clients), t_sec, img_bytes, frame_det, prompt)
        for i, (t_sec, img_bytes, frame_det) in enumerate(inference_tasks)
    ]

    with ThreadPoolExecutor(max_workers=len(clients)) as ex:
        futures = {ex.submit(run_gemini, ta): ta[1] for ta in task_args}
        for future in as_completed(futures):
            t_sec = futures[future]
            try:
                t_sec, frame_detections, analysis = future.result()
                if analysis.get("alert") and "error" not in analysis:
                    alert_seconds.add(t_sec)
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
                            flagged_ids.add(best_tid)
                    print(f"[{t_sec:>5.1f}s] ALERT | {persona['tag']} IDs: {sorted(flagged_ids)}")
                elif "error" not in analysis:
                    print(f"[{t_sec:>5.1f}s] Clear.")
                else:
                    print(f"[{t_sec:>5.1f}s] Gemini error: {analysis.get('error', '?')}")
            except Exception as e:
                print(f"[{t_sec:>5.1f}s] Error: {e}")
    print(f"\nPass 1 complete: {len(raw_tracks)} tracks, {persona['tag']} IDs: {sorted(flagged_ids)}")

    # --------------------------------------------------------
    # PASS 2: RTS smooth + interpolate
    # --------------------------------------------------------

    print("\n========== PASS 2: RTS Smoothing + Interpolation ==========\n")

    smoothed_output: Dict[int, List[Tuple[int, List[int]]]] = {}
    for tid, measurements in raw_tracks.items():
        smoothed = rts_smooth_track(measurements, all_processed_frames)
        if not smoothed:
            continue
        sorted_frames = sorted(smoothed.keys())
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
        last_f = sorted_frames[-1]
        if last_f not in smoothed_output:
            smoothed_output[last_f] = []
        smoothed_output[last_f].append((tid, smoothed[last_f]))

    print(f"Smoothed {len(raw_tracks)} tracks, interpolated to {len(smoothed_output)} frames")

    # --------------------------------------------------------
    # PASS 3: Annotate
    # --------------------------------------------------------

    print("\n========== PASS 3: Annotating ==========\n")

    avi_path = f"{output_name}.avi"
    codecs = [('MJPG', avi_path), ('avc1', f"{output_name}.mp4"), ('mp4v', f"{output_name}.mp4")]
    out = None
    final_path = None
    for codec, path in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if out.isOpened():
            final_path = path
            print(f"Codec {codec} -> {final_path}")
            break

    if out is None or not out.isOpened():
        print("Error: no codec.")
        sys.exit(1)

    GREEN = (0, 200, 0)
    RED = (0, 0, 255)
    tag = persona["tag"]
    alert_label = persona["alert_label"]

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        tracks = smoothed_output.get(frame_idx, [])

        clamped = []
        for tid, box in tracks:
            x1 = max(0, min(box[0], W-1))
            y1 = max(0, min(box[1], H-1))
            x2 = max(x1+1, min(box[2], W))
            y2 = max(y1+1, min(box[3], H))
            clamped.append((tid, [x1, y1, x2, y2]))
        tracks = clamped

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
            is_flagged = tid in flagged_ids
            color = RED if is_flagged else GREEN
            thick = 3 if is_flagged else 2

            if tid in occluded:
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thick)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            label = f"ID {tid}"
            if is_flagged:
                label += f" [{tag}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, max(0, y1-th-6)), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, max(th+2, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if any(tid in flagged_ids for tid, _ in tracks):
            cv2.putText(frame, f"ALERT: {alert_label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

        cv2.putText(frame, f"Time: {t_sec:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 120 == 0:
            print(f"  Annotating: {frame_idx}/{total_frames} ({t_sec:.1f}s)")

    cap.release()
    out.release()

    # Convert to mp4 if we wrote avi
    if final_path.endswith(".avi"):
        mp4_path = f"{output_name}.mp4"
        try:
            import imageio_ffmpeg
            import subprocess
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([exe, '-y', '-i', final_path, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '18', mp4_path],
                           check=True, capture_output=True)
            final_path = mp4_path
            print(f"Converted -> {mp4_path}")
        except Exception as e:
            print(f"FFmpeg conversion failed ({e}), output is {final_path}")

    print(f"\nDone -> {final_path}")

    formatted = sorted([round(s, 1) for s in alert_seconds])
    print(f"Alert timestamps: {formatted}")
    report_path = f"{output_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Persona: {persona['name']}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"{tag} IDs: {sorted(flagged_ids)}\n")
        f.write(f"Alert timestamps: {formatted}\n")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
