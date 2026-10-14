#!/usr/bin/env python3
"""
Assemble translated subtitles into frame-perfect SubRip (.srt) format,
strictly binding English dialogue lines to the millisecond start/end timestamps
from the Japanese speech recognition transcript.
"""

import argparse
import json
import os
import sys

def format_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

def build_srt(transcript_path: str, translation_path: str, output_srt: str):
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file '{transcript_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(translation_path):
        print(f"Error: Translation map '{translation_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    with open(translation_path, "r", encoding="utf-8") as f:
        translations = json.load(f)

    # Convert string keys to int if necessary
    trans_map = {}
    for k, v in translations.items():
        try:
            trans_map[int(k)] = v
        except ValueError:
            trans_map[k] = v

    cues = []
    # If opening theme cues are provided in the translation map under "op_cues", include them
    if "op_cues" in trans_map and isinstance(trans_map["op_cues"], list):
        for cue in trans_map["op_cues"]:
            cues.append((cue["start"], cue["end"], cue["text"]))

    # Map dialogue cues directly to Whisper segment timestamps
    for s in segments:
        sid = s["id"]
        if sid in trans_map:
            text = trans_map[sid]
            if isinstance(text, str) and text.strip():
                cues.append((s["start"], s["end"], text.strip()))

    # Sort chronologically by start timestamp
    cues.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(os.path.abspath(output_srt)), exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as out:
        for idx, (start, end, text) in enumerate(cues, 1):
            out.write(f"{idx}\n")
            out.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            out.write(f"{text}\n\n")

    print(f"Successfully generated {len(cues)} subtitle cues in: {output_srt}")

def main():
    parser = argparse.ArgumentParser(description="Build frame-perfect SRT subtitles from transcript and translations.")
    parser.add_argument("--transcript", "-t", required=True, help="Path to input JSON transcript from transcribe.py")
    parser.add_argument("--translations", "-m", required=True, help="Path to JSON file containing segment translations")
    parser.add_argument("--output", "-o", required=True, help="Path to output .srt file")
    args = parser.parse_args()

    build_srt(args.transcript, args.translations, args.output)

if __name__ == "__main__":
    main()
