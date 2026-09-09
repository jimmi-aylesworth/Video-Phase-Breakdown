#!/usr/bin/env python3
"""
video_phase_breakdown.py

Fully local pipeline for turning a long screen-recorded video (e.g. a security
validation walkthrough) into a structured, phased step-by-step breakdown.

Pipeline:
  1. Extract audio -> transcribe locally with faster-whisper (narration).
  2. Extract frames on scene-change -> describe each with a local Ollama
     vision model (on-screen tool output, terminal state, config, etc).
  3. Merge transcript + frame descriptions into one timestamped timeline.
  4. Feed the timeline to a local Ollama model to synthesize a structured
     phase/step breakdown, written out as Markdown.

Requirements (all local, nothing leaves your machine):
  - ffmpeg installed and on PATH
  - Ollama installed and running (https://ollama.com)
      ollama pull qwen2.5vl:7b      # vision model (default, recommended for
                                     # your RTX 5080 / 16GB VRAM -- strong at
                                     # reading terminal/tool output & UI text)
      # Alternative vision models:
      #   ollama pull minicpm-v          # true multi-frame sequence input
      #   ollama pull llama3.2-vision    # good general OCR, 11B
  - pip install faster-whisper requests

Usage:
  python video_phase_breakdown.py /path/to/video.mp4 \
      --outdir ./validation_run \
      --vision-model qwen2.5vl:7b \
      --whisper-model medium \
      --scene-threshold 0.35

  Resumable: intermediate JSON (transcript, frame descriptions) is cached in
  --outdir, so re-running after a crash / interrupted run skips finished work.
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------------------
# Utility / setup
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_windows_cuda_dll_dirs() -> None:
    """On Windows, pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 land
    their DLLs inside site-packages, but CTranslate2 (used by faster-whisper)
    only searches PATH and standard system DLL locations -- it does not know
    to look there. Scan every site-packages root on disk directly (rather
    than relying on 'nvidia.*' namespace-package import resolution, which is
    unreliable) and register any DLL folders found so 'cublas64_12.dll' etc.
    can actually be located at inference time."""
    if os.name != "nt":
        return

    import site

    roots = set(sys.path)
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.add(site.getusersitepackages())
    except Exception:
        pass

    subpaths = [
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
    ]

    found_any = False
    found_dirs = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for sub in subpaths:
            dll_dir = root_path / sub
            if dll_dir.is_dir():
                try:
                    os.add_dll_directory(str(dll_dir))
                except OSError:
                    pass
                found_dirs.append(str(dll_dir))
                found_any = True

    if found_dirs:
        # CTranslate2 (used by faster-whisper) loads CUDA DLLs with a plain
        # LoadLibrary() call, which does NOT honor AddDllDirectory-registered
        # paths -- it only respects PATH itself. add_dll_directory above is
        # harmless but insufficient on its own; PATH is what actually fixes it.
        os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")
        for d in found_dirs:
            log(f"Registered CUDA DLL directory: {d}")

    if not found_any:
        log(
            "Could not locate nvidia-cublas-cu12/nvidia-cudnn-cu12 DLL folders "
            "under any site-packages root; GPU load may fail and fall back to CPU."
        )


def check_dependencies() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it and re-run.")
    if shutil.which("ffprobe") is None:
        sys.exit("ffprobe not found on PATH (ships with ffmpeg). Install it and re-run.")
    try:
        requests.get(OLLAMA_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        sys.exit(
            "Could not reach Ollama at http://localhost:11434 -- "
            "is `ollama serve` running?"
        )
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        sys.exit("faster-whisper not installed. Run: pip install faster-whisper")


def ensure_model_pulled(model: str) -> None:
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", [])]
    # Ollama tag matching is loose (e.g. "qwen2.5vl:7b" vs "qwen2.5vl:latest")
    if not any(model.split(":")[0] in n for n in names):
        log(f"Model '{model}' not found locally. Pulling it now (one-time)...")
        proc = subprocess.run(["ollama", "pull", model])
        if proc.returncode != 0:
            sys.exit(f"Failed to pull model '{model}'.")


# --------------------------------------------------------------------------
# Step 1: audio extraction + transcription
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path) -> None:
    if out_wav.exists():
        log(f"Audio already extracted -> {out_wav.name}")
        return
    log("Extracting audio track...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_audio(wav_path: Path, cache_path: Path, whisper_model: str) -> list:
    if cache_path.exists():
        log(f"Loading cached transcript -> {cache_path.name}")
        return json.loads(cache_path.read_text())

    log(f"Transcribing with faster-whisper ({whisper_model}, GPU if available)...")
    add_windows_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    # RTX 5080 handles medium/large-v3 comfortably, but on Windows the CUDA
    # path only fails once you actually run inference (CTranslate2 lazy-loads
    # cuBLAS/cuDNN), not at model construction -- so we try the real call and
    # fall back to CPU if that's where it breaks.
    def run_transcribe(device: str, compute_type: str):
        model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        segments, info = model.transcribe(str(wav_path), vad_filter=True)
        return list(segments), info  # materialize the generator inside the try

    try:
        segments, _info = run_transcribe("cuda", "float16")
    except Exception as e:
        log(f"GPU transcription failed ({e}); falling back to CPU (slower).")
        log(
            "To fix GPU accel: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
            "(matching your CUDA version), then re-run."
        )
        segments, _info = run_transcribe("cpu", "int8")
    result = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    cache_path.write_text(json.dumps(result, indent=2))
    log(f"Transcribed {len(result)} segments.")
    return result


# --------------------------------------------------------------------------
# Step 2: scene-change frame extraction + vision descriptions
# --------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path, threshold: float) -> list:
    """Extract frames on scene-change, return sorted list of timestamps (sec)
    aligned with frame_0001.png, frame_0002.png, ..."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("frame_*.png"))
    ts_cache = frames_dir / "timestamps.json"
    if existing and ts_cache.exists():
        log(f"Using {len(existing)} previously extracted frames.")
        return json.loads(ts_cache.read_text())

    log(f"Extracting scene-change frames (threshold={threshold})...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        str(frames_dir / "frame_%04d.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Parse pts_time values out of showinfo stderr output, in order emitted.
    timestamps = [
        float(m) for m in re.findall(r"pts_time:([\d.]+)", proc.stderr)
    ]

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if len(timestamps) != len(frame_files):
        # Fallback: if counts mismatch (can happen on some ffmpeg builds),
        # space frames evenly across the video duration rather than fail.
        log("WARNING: timestamp/frame count mismatch, estimating timestamps.")
        duration = get_video_duration(video_path)
        n = len(frame_files)
        timestamps = [round(i * duration / max(n - 1, 1), 2) for i in range(n)]

    # Always make sure timestamp 0 is represented for context.
    ts_cache.write_text(json.dumps(timestamps, indent=2))
    log(f"Extracted {len(frame_files)} frames.")
    return timestamps


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def describe_frames(
    frames_dir: Path, timestamps: list, cache_path: Path, vision_model: str
) -> list:
    if cache_path.exists():
        log(f"Loading cached frame descriptions -> {cache_path.name}")
        return json.loads(cache_path.read_text())

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    log(f"Describing {len(frame_files)} frames with {vision_model}...")

    prompt = (
        "You are looking at a screenshot from a security validation / "
        "technical walkthrough recording. Describe precisely what is on "
        "screen: application or terminal in focus, any visible commands, "
        "tool output, error/success messages, file paths, or configuration "
        "state. Transcribe any important on-screen text verbatim. Be "
        "concise and factual -- do not speculate about intent."
    )

    results = []
    for i, (frame_path, ts) in enumerate(zip(frame_files, timestamps), 1):
        img_b64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            description = resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            log(f"  Frame {i} failed ({e}), skipping.")
            description = "[description failed]"

        results.append({"timestamp": ts, "description": description})
        if i % 10 == 0 or i == len(frame_files):
            log(f"  {i}/{len(frame_files)} frames described.")

    cache_path.write_text(json.dumps(results, indent=2))
    return results


# --------------------------------------------------------------------------
# Step 3: merge + Step 4: synthesize phase breakdown
# --------------------------------------------------------------------------

def build_timeline(transcript: list, frame_descs: list) -> str:
    events = []
    for seg in transcript:
        events.append((seg["start"], f"[NARRATION {fmt_ts(seg['start'])}] {seg['text']}"))
    for fd in frame_descs:
        events.append(
            (fd["timestamp"], f"[SCREEN {fmt_ts(fd['timestamp'])}] {fd['description']}")
        )
    events.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in events)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def synthesize_breakdown(timeline: str, synthesis_model: str, outdir: Path) -> str:
    log(f"Synthesizing phase breakdown with {synthesis_model}...")

    system_prompt = (
        "You are a security engineer producing formal validation "
        "documentation from a timestamped recording log. The log below "
        "interleaves narration (what the tester said) and screen state "
        "(what was on screen), both timestamped HH:MM:SS.\n\n"
        "Produce a structured Markdown breakdown with:\n"
        "  - Numbered phases (## Phase N: <short title>)\n"
        "  - For each phase: start/end timestamp range, objective, steps "
        "taken (bulleted), tools/commands used, and the observed outcome "
        "(pass/fail/inconclusive if determinable)\n"
        "  - A final '## Summary' section listing all phases in one table: "
        "Phase | Time Range | Outcome\n\n"
        "Only state what is directly supported by the log. If something is "
        "ambiguous, say so rather than guessing."
    )

    # Chunk if the timeline is very long, to stay within context window.
    max_chars = 60000
    if len(timeline) > max_chars:
        log("Timeline is long -- chunking and summarizing hierarchically.")
        chunks = [timeline[i:i + max_chars] for i in range(0, len(timeline), max_chars)]
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            log(f"  Summarizing chunk {i}/{len(chunks)}...")
            partial = call_ollama_chat(
                synthesis_model,
                "Summarize this segment of a timestamped validation log into "
                "a bulleted list of what happened, keeping all timestamps:",
                chunk,
            )
            chunk_summaries.append(partial)
        timeline = "\n\n".join(chunk_summaries)

    breakdown = call_ollama_chat(synthesis_model, system_prompt, timeline)
    return breakdown


def call_ollama_chat(model: str, system_prompt: str, user_content: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--outdir", type=Path, default=Path("./video_analysis_run"))
    parser.add_argument(
        "--vision-model", default="qwen2.5vl:7b",
        help="Ollama vision model (default: qwen2.5vl:7b, recommended for 16GB VRAM)",
    )
    parser.add_argument(
        "--synthesis-model", default=None,
        help="Ollama model for final synthesis (default: same as --vision-model)",
    )
    parser.add_argument(
        "--whisper-model", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper model size (default: medium)",
    )
    parser.add_argument(
        "--scene-threshold", type=float, default=0.35,
        help="ffmpeg scene-change sensitivity, lower = more frames (default: 0.35)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    synthesis_model = args.synthesis_model or args.vision_model

    check_dependencies()
    ensure_model_pulled(args.vision_model)
    if synthesis_model != args.vision_model:
        ensure_model_pulled(synthesis_model)

    args.outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.outdir / "frames"
    wav_path = args.outdir / "audio.wav"

    # Step 1
    extract_audio(args.video, wav_path)
    transcript = transcribe_audio(
        wav_path, args.outdir / "transcript.json", args.whisper_model
    )

    # Step 2
    timestamps = extract_frames(args.video, frames_dir, args.scene_threshold)
    frame_descs = describe_frames(
        frames_dir, timestamps, args.outdir / "frame_descriptions.json", args.vision_model
    )

    # Step 3
    timeline = build_timeline(transcript, frame_descs)
    (args.outdir / "timeline.txt").write_text(timeline)
    log(f"Merged timeline written -> {args.outdir / 'timeline.txt'}")

    # Step 4
    breakdown = synthesize_breakdown(timeline, synthesis_model, args.outdir)
    out_md = args.outdir / "phase_breakdown.md"
    out_md.write_text(breakdown)
    log(f"Done. Phase breakdown written -> {out_md}")


if __name__ == "__main__":
    main()
