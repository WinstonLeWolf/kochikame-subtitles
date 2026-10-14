#!/usr/bin/env python3
"""
Extract audio from a video file into a standardized 16kHz mono audio file (or copy AAC stream)
for speech recognition.
"""

import argparse
import os
import subprocess
import sys

def extract_audio(input_video: str, output_audio: str, copy_stream: bool = False):
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_audio)), exist_ok=True)

    if copy_stream:
        # Lossless fast demuxing without re-encoding
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn",
            "-c:a", "copy",
            output_audio
        ]
    else:
        # Standardized 16kHz mono PCM for Whisper ASR
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            output_audio
        ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: ffmpeg failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Successfully extracted audio to: {output_audio}")

def main():
    parser = argparse.ArgumentParser(description="Extract audio track from anime episode video file.")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file (.mp4, .mkv)")
    parser.add_argument("--output", "-o", required=True, help="Path to output audio file (.wav, .aac, .mp3)")
    parser.add_argument("--copy", action="store_true", help="Copy audio stream directly without re-encoding")
    args = parser.parse_args()

    extract_audio(args.input, args.output, copy_stream=args.copy)

if __name__ == "__main__":
    main()
