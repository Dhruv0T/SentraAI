from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import uvicorn
import os
import shutil
import glob
from typing import Optional

from engine import vision_agent

load_dotenv()

app = FastAPI(title="Vision AI Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PersonaRequest(BaseModel):
    persona: str

class ChatRequest(BaseModel):
    question: str
    report_path: Optional[str] = None  # e.g. "4_parallel_report.txt" (optional, uses latest)

@app.on_event("startup")
async def startup_event():
    try:
        vision_agent.start()
    except Exception as e:
        print(f"Vision agent failed to start: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanly stop the background threads
    vision_agent.stop()

@app.get("/status")
async def get_status():
    """Returns the current active Persona and AI health."""
    return vision_agent.get_status()

@app.post("/switch-persona")
async def switch_persona(request: PersonaRequest):
    """Changes the AI's active role."""
    success = vision_agent.set_persona(request.persona)
    if not success:
        raise HTTPException(status_code=400, detail=f"Persona '{request.persona}' not found in personas.json")
    return {"message": f"Persona switched to {request.persona}", "status": vision_agent.get_status()}

@app.get("/logs")
async def get_logs():
    """Returns a list of the last 50 AI reasoning events."""
    return vision_agent.get_logs()

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a new video file and switches the AI engine to process it."""
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    success = vision_agent.switch_video(file_path)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to switch video source.")
        
    return {"message": f"Successfully switched to {file.filename}", "status": vision_agent.get_status()}

@app.get("/video_feed")
async def video_feed():
    """An MJPEG stream endpoint for the UI."""
    return StreamingResponse(vision_agent.generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

# ---------------------------------------------------------------------------
# Chat about video analysis
# ---------------------------------------------------------------------------

def _get_reports():
    """List *_report.txt files in project directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(base, "*_report.txt")), key=os.path.getmtime, reverse=True)

@app.get("/reports")
async def list_reports():
    """List available analysis reports to chat about."""
    reports = _get_reports()
    return {"reports": [os.path.basename(p) for p in reports]}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask questions about a video analysis. Uses the report file for context.
    If report_path is omitted, uses the most recent report.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    full_path = request.report_path
    base = os.path.dirname(os.path.abspath(__file__))
    if not full_path:
        reports = _get_reports()
        if not reports:
            raise HTTPException(status_code=404, detail="No analysis reports found. Run annotate_4.py first.")
        full_path = reports[0]
    elif not os.path.isabs(full_path):
        full_path = os.path.join(base, full_path)

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Report not found: {full_path}")

    with open(full_path) as f:
        report_content = f.read()

    client = genai.Client(api_key=api_key)
    prompt = f"""You are a helpful assistant that answers questions about video analysis reports.
Here is the analysis report:

---
{report_content}
---

The report contains: Persona used, Video name, flagged/suspect IDs, and alert timestamps (in seconds).
Answer the user's question based on this report. Be concise and factual.

User question: {request.question}
"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "report": os.path.basename(full_path)}

# Chat UI - inline HTML so no static file dependency
CHAT_HTML = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Video Analysis Chat</title>
<style>body{font-family:system-ui;max-width:600px;margin:2rem auto;padding:0 1rem}
#messages{border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:200px;max-height:400px;overflow-y:auto;background:#f9f9f9}
.msg{margin-bottom:.75rem}.msg.user{color:#06c}.msg.bot{color:#333}
form{display:flex;gap:.5rem;margin-top:.5rem}input[type="text"]{flex:1;padding:.5rem}
button{padding:.5rem 1rem;background:#06c;color:#fff;border:none;border-radius:6px;cursor:pointer}
</style></head><body>
<h1>Video Analysis Chat</h1><p>Ask questions about what was flagged.</p>
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

@app.get("/", response_class=HTMLResponse)
async def root():
    return CHAT_HTML

@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui():
    return CHAT_HTML

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
