import cv2
import time
import threading
import base64
import json
import os
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Define the expected structured output schema using Pydantic
class VisionAnalysis(BaseModel):
    alert: bool = Field(description="Whether the AI has detected an alert/violation based on the persona.")
    reasoning: str = Field(description="Detailed explanation of what the AI observed and why it made its decision.")
    detected_objects: List[str] = Field(description="List of relevant objects detected in the scene.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")

class VisionAgent:
    def __init__(self, personas_file: str = "personas.json"):
        # Load personas
        with open(personas_file, 'r') as f:
            self.personas: Dict[str, str] = json.load(f)
        
        # Initial State
        self.active_persona_name: str = list(self.personas.keys())[0]
        self.latest_frame = None
        self.logs: List[Dict] = []
        self.max_logs: int = 50
        
        # Threading control
        self.running: bool = False
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.video_path: str = "sample_video.mp4"
        self.cap = None
        
        # Gemini Client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY environment variable not set. API calls will fail.")
            api_key = "dummy_key"
        self.client = genai.Client(api_key=api_key)

    def start(self):
        """Starts the background capture and inference threads."""
        if self.running:
            return
        self.running = True
        
        if not getattr(self, 'cap', None) or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("VisionAgent started.")
        
    def stop(self):
        """Stops the background threads."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
        print("VisionAgent stopped.")

    def set_persona(self, persona_name: str) -> bool:
        """Updates the active persona."""
        if persona_name in self.personas:
            self.active_persona_name = persona_name
            return True
        return False

    def get_status(self) -> Dict:
        """Returns the current engine status."""
        return {
            "active_persona": self.active_persona_name,
            "capture_thread_alive": self.capture_thread.is_alive() if self.capture_thread else False,
            "inference_thread_alive": self.inference_thread.is_alive() if self.inference_thread else False,
            "frame_available": self.latest_frame is not None,
            "api_key_configured": bool(os.environ.get("GEMINI_API_KEY"))
        }

    def get_logs(self) -> List[Dict]:
        """Returns the most recent reasoning logs."""
        return self.logs

    def switch_video(self, video_path: str) -> bool:
        """Stops the current capture and restarts it with a new video file."""
        if not os.path.exists(video_path):
            return False
            
        self.video_path = video_path
        
        # Stop threads temporarily
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
            
        if getattr(self, 'cap', None):
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        self.latest_frame = None
        
        # Restart the threads safely
        self.start()
            
        return True

    def _capture_loop(self):
        """
        Background thread for video file processing. 
        Continuously reads from the video file.
        """
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}.")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Update the shared state with the latest frame
                self.latest_frame = frame
                # Artificial delay to simulate real-time playback speed (e.g. 30fps)
                time.sleep(1/30.0)
            else:
                # Video ended, loop back to the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        self.cap.release()

    def _inference_loop(self):
        """
        Background thread for AI inference.
        Grabs the latest frame every 2 seconds and queries Gemini.
        """
        model_id = 'gemini-2.5-flash'
        while self.running:
            start_time = time.time()
            frame = self.latest_frame
            persona_prompt = self.personas.get(self.active_persona_name, "Analyze the scene.")
            
            if frame is not None and os.environ.get("GEMINI_API_KEY"):
                try:
                    # Encode frame to base64 jpeg
                    _, buffer = cv2.imencode('.jpg', frame)
                    encoded_image = base64.b64encode(buffer).decode('utf-8')
                    
                    image_part = types.Part.from_bytes(
                        data=base64.b64decode(encoded_image),
                        mime_type='image/jpeg',
                    )
                    
                    prompt = f"System Persona: {persona_prompt}\n\nTask: Analyze this image and provide your reasoning in the requested JSON format."
                    
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=[prompt, image_part],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=VisionAnalysis,
                        ),
                    )
                    
                    # Parse the JSON response
                    try:
                        analysis_result = json.loads(response.text)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON response: {response.text}")
                        analysis_result = {"error": "Invalid JSON response from AI"}

                    log_entry = {
                        "timestamp": time.time(),
                        "persona": self.active_persona_name,
                        "analysis": analysis_result
                    }
                    
                    self.logs.insert(0, log_entry)
                    if len(self.logs) > self.max_logs:
                        self.logs.pop()
                        
                except Exception as e:
                    print(f"Inference error: {e}")
            
            # Wait until 2 seconds have passed since the start of the loop
            elapsed = time.time() - start_time
            sleep_time = max(0, 2.0 - elapsed)
            time.sleep(sleep_time)

    def generate_video_feed(self):
        """
        Generator for MJPEG streaming. Yields the latest frame boundary.
        """
        while self.running:
            frame = self.latest_frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Yield at roughly 30fps
            time.sleep(1/30.0)

# Create a singleton instance for the FastAPI app to use
vision_agent = VisionAgent()
