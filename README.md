# Sentra

**AI-powered video analysis for surveillance and security.** Sentra analyzes video footage to detect fights, suspicious activity, shoplifting, and other security events using YOLO object detection and Gemini AI.

## Project Description

Sentra processes surveillance and retail video to flag concerning behavior in real time. Upload a video, run analysis with configurable personas (fight detection, store security, shoplifting), and chat with an AI assistant about what was flagged—alert timestamps, suspect IDs, and summaries.

## Features

- **Video upload** — Upload videos via the web UI
- **Multi-persona analysis** — Fight detection, store security, shoplifting
- **Annotated outputs** — Videos with bounding boxes and alert overlays
- **AI chat** — Ask questions about reports, flagged times, and summaries
- **Reports** — Per-video analysis reports for audit and review

## Screenshots

| Home — Upload & recent videos | Home — Video cards | Chat — AI reports |
|:---:|:---:|:---:|
| ![Home page](page%201%20%28home%29.png) | ![Home cards](page%202%20%28home%29.png) | ![Chat](page%203%20%28chat%29.png) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
cp .env.example .env
# Edit .env and add GEMINI_API_KEY=your_key

# Run the server
uvicorn main_chat:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Analysis

To analyze a video and generate a report:

```bash
python annotate_4.py -v your_video.mp4
```

This produces an annotated video and a report file. Use the Chat page to ask questions about the report.
