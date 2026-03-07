import time
import sys
from dotenv import load_dotenv

def test():
    load_dotenv()
    from engine import VisionAgent
    
    video_path = "4.mp4"
    print(f"Testing VisionAgent with video: {video_path}")
    
    agent = VisionAgent()
    # Before starting, check if switch video works
    success = agent.switch_video(video_path)
    print(f"Switch video success: {success}")
    if not success:
        print("Failed to switch video.")
        sys.exit(1)

    print("Agent started. Waiting 15 seconds to collect inference logs...")
    time.sleep(15)
    
    logs = agent.get_logs()
    print(f"Collected {len(logs)} logs.")
    for i, log in enumerate(logs):
        print(f"Log {i+1}: timestamp={log.get('timestamp')}, persona={log.get('persona')}")
        print(f"  Analysis: {log.get('analysis')}")
        
    print("Stopping agent...")
    agent.stop()
    print("Test finished.")

if __name__ == "__main__":
    test()
