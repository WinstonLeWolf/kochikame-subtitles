#!/usr/bin/env python3
"""
Transcribe Japanese anime audio into timed Japanese speech segments using faster-whisper.
CRITICAL: Runs with language='ja' and task='transcribe' (NOT 'translate') to preserve
exact native Japanese speech, grammar, and millisecond timestamps without AI hallucination.
"""

import argparse
import json
import os
import sys
import time

def setup_encoding():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

def transcribe_audio(
    audio_path: str,
    output_json: str,
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    threads: int = 8
):
    setup_encoding()

    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: 'faster-whisper' is not installed. Run: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=threads)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Domain-specific prompt to guide ASR on character names and terminology
    initial_prompt = (
        "こちら葛飾区亀有公園前派出所。両津勘吉、大原部長、中川圭一、秋本・カトリーヌ・麗子、"
        "本田速人、寺井洋介。下町、交番、葛飾署、始末書。"
    )

    print(f"Transcribing {audio_path} in native Japanese (task='transcribe')...")
    t1 = time.time()

    segments, info = model.transcribe(
        audio_path,
        language="ja",
        task="transcribe",
        initial_prompt=initial_prompt,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        word_timestamps=False
    )

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    results = []
    count = 0
    for segment in segments:
        count += 1
        results.append({
            "id": count,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip()
        })
        if count % 25 == 0:
            print(f"[{segment.start:6.1f}s -> {segment.end:6.1f}s] ({count} segments) {segment.text.strip()}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Extracted {len(results)} segments in {time.time() - t1:.1f}s.")
    print(f"Saved transcript to: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe Japanese anime audio with faster-whisper.")
    parser.add_argument("--input", "-i", required=True, help="Path to input audio file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON transcript")
    parser.add_argument("--model", "-m", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute-type", default="int8", help="Quantization (int8, float16, float32)")
    parser.add_argument("--threads", "-t", type=int, default=os.cpu_count() or 4, help="CPU threads to utilize")
    args = parser.parse_args()

    transcribe_audio(
        args.input,
        args.output,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        threads=args.threads
    )

if __name__ == "__main__":
    main()
