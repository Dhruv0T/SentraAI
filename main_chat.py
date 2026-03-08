"""
Minimal server with ONLY chat + reports. No video/OpenCV - use this if main.py crashes.
Run: uvicorn main_chat:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import os
import glob
import re
from typing import Optional, List, Dict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

load_dotenv()

app = FastAPI(title="Vision AI Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    report_path: Optional[str] = None

def _get_reports():
    base = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(base, "*_report.txt")), key=os.path.getmtime, reverse=True)

def _video_from_report(report_path: str) -> Optional[str]:
    """Extract video filename from report content. Returns path to video file if it exists."""
    base = os.path.dirname(report_path)
    candidates = []
    try:
        with open(report_path) as f:
            for line in f:
                m = re.search(r"Video:\s*(.+)", line, re.I)
                if m:
                    vid = m.group(1).strip()
                    if vid:
                        candidates.append(os.path.join(base, vid))
                    break
    except Exception:
        pass
    # Fallback: derive from report filename (e.g. new0annoated_report.txt -> new0annoated.mp4)
    stem = os.path.basename(report_path).replace("_report.txt", "").replace("_annotated", "").replace("_annoated", "").replace("_anno", "")
    if stem:
        for suffix in ["_annotated.mp4", "_annoated.mp4", ".mp4"]:
            candidates.append(os.path.join(base, stem + suffix))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None

def _extract_thumbnail(video_path: str) -> Optional[bytes]:
    """Extract a JPEG frame from video. Requires cv2."""
    if not HAS_CV2:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        # Get frame at 10% into video (skip black intro)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = max(0, int(total * 0.1)) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
    except Exception:
        return None

CHAT_HTML = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Video Analysis Chat</title>
<style>body{font-family:system-ui;max-width:600px;margin:2rem auto;padding:0 1rem}
#messages{border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:200px;max-height:400px;overflow-y:auto;background:#f9f9f9}
.msg{margin-bottom:.75rem}.msg.user{color:#06c}.msg.bot{color:#333}
form{display:flex;gap:.5rem;margin-top:.5rem}input[type="text"]{flex:1;padding:.5rem}
button{padding:.5rem 1rem;background:#06c;color:#fff;border:none;border-radius:6px;cursor:pointer}
</style></head><body>
<h1>Video Analysis Chat</h1><p>Ask questions about what was flagged in your videos.</p>
<select id="report"></select>
<div id="messages"></div>
<form id="form"><input type="text" id="input" placeholder="e.g. What times were people flagged?"/><button type="submit">Send</button></form>
<script>
const API=location.origin;
const msgEl=document.getElementById('messages'),formEl=document.getElementById('form'),inputEl=document.getElementById('input'),reportEl=document.getElementById('report');
async function loadReports(){try{const r=await fetch(API+'/reports');const d=await r.json();reportEl.innerHTML=d.reports.map(f=>'<option value="'+f+'">'+f+'</option>').join('')}catch(e){reportEl.innerHTML='<option>No reports</option>'}}
function addMsg(t,u){const d=document.createElement('div');d.className='msg '+(u?'user':'bot');d.innerHTML=u?t:'<strong>Assistant:</strong> '+t;msgEl.appendChild(d);msgEl.scrollTop=msgEl.scrollHeight}
formEl.onsubmit=async e=>{e.preventDefault();const q=inputEl.value.trim();if(!q)return;addMsg(q,true);inputEl.value='';
const body={question:q};if(reportEl.value&&reportEl.value!='No reports')body.report_path=reportEl.value;
try{const r=await fetch(API+'/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();
addMsg(r.ok?d.answer:'Error: '+(d.detail||''),false)}catch(e){addMsg('Error: '+e.message,false)}}
loadReports();
</script></body></html>'''

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

@app.get("/", response_class=HTMLResponse)
async def root():
    p = os.path.join(_FRONTEND, "index.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return CHAT_HTML

@app.get("/chat", response_class=HTMLResponse)
@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui():
    p = os.path.join(_FRONTEND, "chat.html")
    if os.path.isfile(p):
        return FileResponse(p)
    return CHAT_HTML

@app.get("/reports")
async def list_reports():
    reports = _get_reports()
    base = os.path.dirname(os.path.abspath(__file__))
    items = []
    for p in reports:
        name = os.path.basename(p)
        display = name.replace("_report.txt", "").replace("_", " ").replace("annotated", "").strip()
        video_path = _video_from_report(p)
        items.append({
            "name": name,
            "display_name": display or name,
            "has_thumbnail": video_path is not None and HAS_CV2,
        })
    return {"reports": [os.path.basename(p) for p in reports], "items": items}

@app.get("/thumbnail/{report_name:path}")
async def get_thumbnail(report_name: str):
    """Return a JPEG thumbnail extracted from the video associated with this report."""
    base = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base, report_name)
    if not report_path.endswith("_report.txt"):
        report_path = report_name
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    video_path = _video_from_report(report_path)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found for this report")
    jpg = _extract_thumbnail(video_path)
    if not jpg:
        raise HTTPException(status_code=404, detail="Could not extract thumbnail")
    return Response(content=jpg, media_type="image/jpeg")

@app.post("/chat")
async def chat(request: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    base = os.path.dirname(os.path.abspath(__file__))
    full_path = request.report_path
    if not full_path:
        reports = _get_reports()
        if not reports:
            raise HTTPException(status_code=404, detail="No reports found. Run annotate_4.py first.")
        full_path = reports[0]
    elif not os.path.isabs(full_path):
        full_path = os.path.join(base, full_path)

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Report not found: {full_path}")

    with open(full_path) as f:
        report_content = f.read()

    client = genai.Client(api_key=api_key)
    prompt = f"""You answer questions about video analysis reports.
Report:
---
{report_content}
---
User question: {request.question}"""

    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "report": os.path.basename(full_path)}

if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_chat:app", host="0.0.0.0", port=8000)
