# Video-Phase-Breakdown
A pythonic pipeline for processing training videos, breaking them down into phases. Useful for when there are no manuals provided with learning content.

---

# Setup Guide (Windows)

Full setup for `video_phase_breakdown.py`, from a bare Windows machine to a working GPU-accelerated run. Tested config: i9 / 64GB RAM / RTX 5080 (16GB VRAM).

## 1. Prerequisites

- **Python** 3.10+ installed and on PATH (`python --version` to confirm)
- **NVIDIA driver** installed (`nvidia-smi` should show your GPU and a CUDA version at the top — driver is backward-compatible, so it just needs to report CUDA 12.x or higher)
- **ffmpeg** installed and on PATH — confirm with `ffmpeg -version` and `ffprobe -version` in the same terminal you'll run the script from. If not on PATH, run `winget install -e --id Gyan.FFmpeg` to install.
- **Ollama** installed — https://ollama.com. Confirm it's running with `ollama list` (starts the background service if not already running; or run `ollama serve` manually).

## 2. Python packages

```powershell
pip install faster-whisper requests
```

## 3. GPU acceleration for transcription (faster-whisper)

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

That's it on the package side — **no manual PATH editing needed**. The script itself (`add_windows_cuda_dll_dirs()`) automatically finds wherever pip put these DLLs under `site-packages\nvidia\...\bin` and prepends them to the process's `PATH` at runtime, which is what CTranslate2 (faster-whisper's backend) actually needs — it uses a plain `LoadLibrary()` call that only respects `PATH`, not the newer `os.add_dll_directory()` mechanism.

**If GPU transcription still fails after this**, the next most likely gap: cuDNN 9 on Windows depends on `zlibwapi.dll`, which NVIDIA does not ship in the pip wheel. If the error message changes to reference that DLL specifically:
1. Download `zlibwapi.dll` from NVIDIA's zlib page (search "zlibwapi.dll NVIDIA cudnn")
2. Drop it in `C:\Windows\System32\` (or any folder already on PATH)

If GPU transcription fails for any reason, the script automatically falls back to CPU — it'll still work, it's just piss slow.

## 4. Ollama vision model

```powershell
ollama pull qwen2.5vl:7b
```

(This is what the script defaults to via `--vision-model`. Fits well within 16GB VRAM and is strong at reading on-screen text/terminal output, which matters for security-validation-style recordings.)

## 5. The script itself

Save `video_phase_breakdown.py` locally (e.g. `C:\Users\<you>\Documents\code\VPB\`).

## 6. Running it

```powershell
python .\video_phase_breakdown.py "C:\path\to\video.mp4" --outdir "C:\path\to\output_folder"
```

On a working run you should see, near the top of the output:
```
Registered CUDA DLL directory: ...\nvidia\cublas\bin
Registered CUDA DLL directory: ...\nvidia\cudnn\bin
Registered CUDA DLL directory: ...\nvidia\cuda_runtime\bin
Transcribing with faster-whisper (medium, GPU if available)...
```
with no fallback-to-CPU message following — that confirms GPU transcription is active.

## Quick sanity checklist (paste-and-check order)

```powershell
python --version
nvidia-smi
ffmpeg -version
ffprobe -version
ollama list
pip show faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If all five come back clean, you're ready to rock!
