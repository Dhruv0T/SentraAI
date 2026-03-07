from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os
import shutil

from engine import vision_agent

load_dotenv()

app = FastAPI(title="Vision AI Backend", version="1.0.0")

class PersonaRequest(BaseModel):
    persona: str

@app.on_event("startup")
async def startup_event():
    # Start the background threads when FastAPI starts
    vision_agent.start()

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
