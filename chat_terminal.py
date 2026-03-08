#!/usr/bin/env python3
"""
Terminal chat about video analysis. Ask questions or get a summary of suspicious activity.
Usage:
  python chat_terminal.py                    # interactive
  python chat_terminal.py --summary          # one-shot summary
  python chat_terminal.py --report 4_parallel_report.txt
"""
import argparse
import glob
import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_reports():
    base = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        glob.glob(os.path.join(base, "*_report.txt")),
        key=os.path.getmtime,
        reverse=True,
    )


def chat(report_path: str, question: str) -> str:
    with open(report_path) as f:
        report = f.read()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    prompt = f"""You are an assistant that summarizes video analysis reports.
Report:
---
{report}
---
Answer the user's question. For summaries, focus on: what was flagged, when (timestamps), how many people, and a brief narrative of the suspicious activity.

User: {question}"""

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text


def main():
    parser = argparse.ArgumentParser(description="Chat about video analysis in terminal")
    parser.add_argument("--report", "-r", default=None, help="Report file (default: latest)")
    parser.add_argument("--summary", "-s", action="store_true", help="Print a summary of suspicious activity and exit")
    args = parser.parse_args()

    reports = get_reports()
    if not reports:
        print("No *_report.txt files found. Run annotate_4.py first.")
        sys.exit(1)

    report_path = args.report
    if report_path and not os.path.isabs(report_path):
        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), report_path)
    if not report_path or not os.path.isfile(report_path):
        report_path = reports[0]

    print(f"Using report: {os.path.basename(report_path)}\n")

    if args.summary:
        answer = chat(report_path, "Give a concise summary of the suspicious activity in this video: what was flagged, at what times, and how many people were involved.")
        print(answer)
        return

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        print("Assistant: ", end="", flush=True)
        try:
            answer = chat(report_path, q)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
