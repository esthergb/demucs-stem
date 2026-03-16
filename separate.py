#!/usr/bin/env python3
"""
Stem separator using Demucs htdemucs_ft model.

Usage:
    python separate.py song.flac
    python separate.py song.mp3
    python separate.py /path/to/music/directory
    python separate.py song.flac --device cpu
    python separate.py song.flac --shifts 5

    # As compiled binary:
    ./demucs-stem song.flac
    ./demucs-stem /path/to/music/directory

Output is written next to the original file:
    song.flac -> song_drums.wav, song_bass.wav, song_voices.wav, song_other.wav
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# PyInstaller bundle compatibility patches
if getattr(sys, "_MEIPASS", None):
    warnings.filterwarnings("ignore", message="Unable to retrieve source")
    warnings.filterwarnings("ignore", message="resource_tracker")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.audio import convert_audio, prevent_clip
from demucs.pretrained import get_model

MODEL_NAME = "htdemucs_ft"

SUPPORTED_EXTENSIONS = {".flac", ".mp3", ".wav", ".ogg", ".aiff", ".aif"}

# Demucs uses "vocals" internally, we rename to match user expectation
STEM_DISPLAY_NAMES = {
    "drums": "drums",
    "bass": "bass",
    "vocals": "voices",
    "other": "other",
}


def get_bundled_model_repo() -> Path | None:
    """Return path to embedded models when running as a PyInstaller binary."""
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        return None
    repo = Path(base) / "models"
    if repo.is_dir():
        return repo
    return None


def parse_args() -> argparse.Namespace:
    # Strip sys.argv of any flags injected by dora/hydra when running
    # inside a PyInstaller bundle (they add -B -S -I -c flags)
    if getattr(sys, "_MEIPASS", None):
        clean_argv = [sys.argv[0]]
        skip_next = False
        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg == "-c":
                skip_next = True
                continue
            if arg in ("-B", "-S", "-I"):
                continue
            clean_argv.append(arg)
        sys.argv = clean_argv

    ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    parser = argparse.ArgumentParser(
        description="Separate audio into stems using htdemucs_ft"
    )
    parser.add_argument(
        "input", type=Path,
        help=f"Input audio file ({ext_list}) or directory",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'mps', 'cuda', 'cpu' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--shifts", type=int, default=1,
        help="Number of random shifts for better quality (default: 1, try 5 for best)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.25,
        help="Overlap between segments (default: 0.25)",
    )
    parser.add_argument(
        "--segment", type=int, default=None,
        help="Segment length in seconds (default: model default). Lower = less memory",
    )
    parser.add_argument(
        "--clip-mode", choices=["clamp", "rescale", "none"], default="clamp",
        help="Clipping strategy (default: clamp)",
    )
    parser.add_argument(
        "--float32", action="store_true",
        help="Save as 32-bit float WAV instead of 16-bit int",
    )
    return parser.parse_args()


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_audio_files(path: Path) -> list[Path]:
    """Return a list of supported audio files from a file path or directory."""
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Warning: unsupported format {path.suffix}, trying anyway...")
        return [path]
    if path.is_dir():
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        files.sort()
        if not files:
            print(f"Error: no audio files found in {path}")
            print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            sys.exit(1)
        return files
    print(f"Error: path not found: {path}")
    sys.exit(1)


def _patch_torch_load():
    """Patch demucs.states.load_model for PyTorch >= 2.6 compatibility.

    Demucs checkpoints serialize full model classes (not just state_dicts),
    which requires weights_only=False in torch.load(). Since PyTorch 2.6
    defaults to weights_only=True, we patch the load call in demucs.states.
    """
    import demucs.states as _states
    import functools

    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    _states.torch.load = _patched_load


def load_model(device: str):
    """Load htdemucs_ft, using embedded weights if running as binary."""
    _patch_torch_load()

    repo = get_bundled_model_repo()
    if repo:
        print("Loading model from embedded weights...")
        model = get_model(MODEL_NAME, repo=repo)
    else:
        print("Loading model (downloading if needed)...")
        model = get_model(MODEL_NAME)
    model.to(device)
    return model


def save_stem_wav(
    wav: torch.Tensor,
    path: Path,
    samplerate: int,
    clip: str = "clamp",
    as_float: bool = False,
) -> None:
    """Save a stem tensor as WAV using soundfile (no torchaudio dependency).

    Args:
        wav: Tensor of shape (channels, samples)
        path: Output file path
        samplerate: Sample rate in Hz
        clip: Clipping strategy ('clamp', 'rescale', 'none')
        as_float: If True, save as 32-bit float; otherwise 16-bit PCM
    """
    wav = prevent_clip(wav, mode=clip)
    # soundfile expects (samples, channels) numpy array
    audio_np = wav.numpy().T
    subtype = "FLOAT" if as_float else "PCM_16"
    sf.write(str(path), audio_np, samplerate, subtype=subtype)


def separate_file(
    file: Path, model, device: str, args: argparse.Namespace,
) -> None:
    """Separate a single audio file and save stems next to the original."""
    print(f"\n{'─' * 60}")
    print(f"Separating: {file}")

    # soundfile returns (samples, channels) as float32 numpy array
    audio_np, sr = sf.read(str(file), dtype="float32")
    # Handle mono files: soundfile returns (samples,) for mono
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=-1)
    # Convert to torch tensor with shape (channels, samples) as demucs expects
    wav = torch.from_numpy(audio_np.T)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    mix = wav.unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        sources = apply_model(
            model, mix,
            shifts=args.shifts,
            split=True,
            overlap=args.overlap,
            progress=True,
            device=device,
            segment=args.segment,
        )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # sources shape: (batch, num_sources, channels, samples) -> remove batch
    sources = sources[0]

    # Save stems next to the original file: {name}_{stem}.wav
    for i, stem_name in enumerate(model.sources):
        display_name = STEM_DISPLAY_NAMES.get(stem_name, stem_name)
        out_path = file.parent / f"{file.stem}_{display_name}.wav"
        save_stem_wav(
            sources[i].cpu(),
            out_path,
            samplerate=model.samplerate,
            clip=args.clip_mode,
            as_float=args.float32,
        )
        print(f"  {out_path.name}")


def main():
    args = parse_args()
    files = collect_audio_files(args.input)
    device = detect_device(args.device)

    print(f"Device:  {device}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Files:   {len(files)}")

    model = load_model(device)
    print(f"Sources: {', '.join(STEM_DISPLAY_NAMES[s] for s in model.sources)}")

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        separate_file(file, model, device, args)

    print(f"\n{'─' * 60}")
    print(f"All done. {len(files)} file(s) processed.")


if __name__ == "__main__":
    # Required for PyInstaller: prevents multiprocessing subprocess from
    # re-executing main() and crashing on sys.argv parsing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
