# demucs-stem

A self-contained audio stem separator powered by Meta's [Demucs](https://github.com/facebookresearch/demucs) `htdemucs_ft` model. Splits any audio file into four stems: **drums**, **bass**, **voices**, and **other**.

Works as a Python script or as a standalone binary (no Python required, fully offline).

## Features

- **4-stem separation** — drums, bass, voices, other
- **Offline binary** — model weights (~320 MB) embedded in the executable
- **Batch processing** — pass a single file or a directory to process all audio files recursively
- **Multiple formats** — `.flac`, `.mp3`, `.wav`, `.ogg`, `.aiff`
- **Apple Silicon acceleration** — auto-detects MPS on macOS; also supports CUDA and CPU
- **Output alongside originals** — stems are saved next to the source file as `{name}_{stem}.wav`

## Quick Start

```bash
# Single file
./demucs-stem song.flac

# Entire directory (recursive)
./demucs-stem /path/to/music/

# Higher quality (5x slower)
./demucs-stem song.flac --shifts 5

# Force CPU
./demucs-stem song.flac --device cpu
```

Output for `song.flac`:
```
song_drums.wav
song_bass.wav
song_voices.wav
song_other.wav
```

## Usage

```
usage: demucs-stem [-h] [--device DEVICE] [--shifts SHIFTS]
                   [--overlap OVERLAP] [--segment SEGMENT]
                   [--clip-mode {clamp,rescale,none}] [--float32]
                   input

Separate audio into stems using htdemucs_ft

positional arguments:
  input                 Input audio file or directory containing audio files

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device: 'mps', 'cuda', 'cpu' (auto-detected if omitted)
  --shifts SHIFTS       Random shifts for better quality (default: 1, try 5 for best)
  --overlap OVERLAP     Overlap between segments (default: 0.25)
  --segment SEGMENT     Segment length in seconds (default: model default). Lower = less memory
  --clip-mode           Clipping strategy: clamp, rescale, none (default: clamp)
  --float32             Save as 32-bit float WAV instead of 16-bit PCM
```

### Options Explained

| Option | Default | Description |
|---|---|---|
| `--shifts N` | `1` | Number of random time-shifts averaged together. Higher = better quality but linearly slower. `5` is a good quality/speed trade-off. |
| `--overlap` | `0.25` | Overlap ratio between processing segments (0.0–1.0). Higher reduces boundary artifacts. |
| `--segment` | model default | Segment length in seconds. Lower values reduce peak memory usage at the cost of quality. Useful on machines with limited RAM/VRAM. |
| `--clip-mode` | `clamp` | How to handle clipping: `clamp` hard-clips to [-1, 1], `rescale` normalizes the mix to avoid clipping, `none` does nothing. |
| `--float32` | off | Save stems as 32-bit float WAV instead of 16-bit integer. Useful for further processing in a DAW. |

## Performance

Benchmarked on Apple M4 Pro (MPS) with a ~4:17 track:

| Shifts | Time | Quality |
|---|---|---|
| 1 | ~100s | Good |
| 2 | ~200s | Better |
| 5 | ~500s | Best (recommended for final masters) |

Processing time scales linearly with `--shifts` and track duration. CPU mode is roughly 3-5x slower than MPS/CUDA.

## Building from Source

### Prerequisites

- Python 3.12+
- ~2 GB disk space for the virtual environment
- ~500 MB for the final binary

### 1. Set Up the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio demucs soundfile numpy tqdm pyinstaller
```

### 2. Download Model Weights

The `htdemucs_ft` model is an ensemble of 4 fine-tuned Hybrid Transformer Demucs models. Download the weights into a `models/` directory:

```bash
mkdir -p models

# Download the 4 model checkpoints + config
curl -L -o models/htdemucs_ft.yaml \
  "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_ft.yaml"

# Parse checkpoint filenames from the YAML and download each one
grep -oE '[0-9a-f]{8}-[0-9a-f]{8}\.th' models/htdemucs_ft.yaml | while read f; do
  curl -L -o "models/$f" "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/$f"
done
```

Each checkpoint is ~80 MB (4 × 80 MB = ~320 MB total).

> **Alternatively**, you can run the script once without a binary (`python separate.py song.flac`) and the model will be downloaded automatically to `~/.cache/torch/hub/checkpoints/`. You can then copy those files into `models/`.

### 3. Build the Binary

```bash
source .venv/bin/activate
pyinstaller demucs-stem.spec
```

This produces a single self-contained executable at `dist/demucs-stem` (~400 MB). The binary embeds:

- Python interpreter
- PyTorch runtime (MPS/CPU backends)
- Demucs library + model weights
- libsndfile (audio I/O)

Build takes ~2-3 minutes.

### 4. Test the Binary

```bash
./dist/demucs-stem song.flac
```

You should see the model loading from embedded weights (no network access needed).

## Project Structure

```
stemSeparator/
├── separate.py          # Main separation script
├── demucs-stem.spec     # PyInstaller build spec
├── pyi_rthook_torch.py  # Runtime hook for PyTorch compatibility
├── models/              # Model weights (not in git, built locally)
│   ├── htdemucs_ft.yaml
│   ├── 04573f0d-f3cf25b2.th
│   ├── 92cfc3b6-ef3bcb9c.th
│   ├── d12395a8-e57c48e6.th
│   └── f7e0c4bc-ba3fe64a.th
├── dist/
│   └── demucs-stem      # Compiled binary (~400 MB)
├── build/               # PyInstaller build artifacts
└── .venv/               # Python virtual environment
```

### Key Files

- **`separate.py`** — Core logic: audio loading (via soundfile/libsndfile), model inference (via Demucs/PyTorch), and WAV output. Handles both script and binary execution modes.
- **`demucs-stem.spec`** — PyInstaller configuration. Bundles model weights from `models/`, demucs remote configs, and libsndfile. Excludes torchcodec/FFmpeg to avoid dynamic library issues.
- **`pyi_rthook_torch.py`** — Runtime hook that patches `inspect.getsource()` to return empty string instead of crashing when PyTorch tries to read source files that don't exist inside the frozen bundle.

## How It Works

1. **Audio loading** — Uses `soundfile` (backed by libsndfile) instead of torchaudio/torchcodec to avoid FFmpeg dependency issues in the binary.
2. **Resampling** — Demucs' `convert_audio()` resamples to 44.1 kHz stereo (what the model expects).
3. **Separation** — `apply_model()` runs the input through 4 Hybrid Transformer Demucs models and averages their predictions. Each model processes the audio in overlapping segments.
4. **Clipping** — `prevent_clip()` ensures output samples stay within [-1, 1].
5. **Output** — Each stem is saved as a 16-bit PCM WAV (or 32-bit float with `--float32`) next to the original file.

### Why soundfile instead of torchaudio?

torchaudio 2.10+ requires torchcodec, which depends on FFmpeg shared libraries (`.dylib`). These are notoriously difficult to bundle in PyInstaller binaries due to dynamic linking and codec licensing. `soundfile` uses libsndfile, which is a single self-contained C library with no external dependencies — much easier to embed.

### Why the `pyi_rthook_torch.py` hook?

PyTorch 2.10 calls `inspect.getsource()` at import time to parse `@compile_ignored` decorator comments. Inside a PyInstaller bundle, `.py` source files don't exist — only compiled bytecode. Without this hook, `import torch` crashes immediately with an `OSError`.

## Troubleshooting

### `MPS backend out of memory`

Use `--segment` to reduce memory usage:

```bash
./demucs-stem song.flac --segment 30
```

This processes 30-second chunks instead of the default (~40s). Lower values use less memory but may slightly reduce quality at segment boundaries.

### Binary crashes on launch

If the binary crashes silently, run it from a terminal to see error output. Common causes:

- **macOS Gatekeeper**: Run `xattr -cr dist/demucs-stem` to clear quarantine flags.
- **Missing model weights**: Rebuild with `pyinstaller demucs-stem.spec` after ensuring `models/` contains all 4 `.th` files and the `.yaml`.

### Slow performance on macOS

Ensure MPS is being used (you should see `Device: mps` in the output). If it falls back to CPU, check that you have macOS 12.3+ and a supported Apple Silicon or AMD GPU.

## License

MIT License. See [LICENSE](LICENSE) for details.

### Third-Party Licenses

This project bundles or depends on the following third-party software:

| Component | License | Usage |
|---|---|---|
| [Demucs](https://github.com/facebookresearch/demucs) (Meta Research) | MIT | Audio source separation model and library |
| [PyTorch](https://pytorch.org) | BSD-3-Clause | Deep learning runtime |
| [NumPy](https://numpy.org) | BSD-3-Clause | Numerical computing |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | Audio I/O Python bindings |
| [libsndfile](http://www.mega-nerd.com/libsndfile/) | **LGPL-2.1** | Audio I/O C library (bundled as shared library in binary) |
| [julius](https://github.com/adefossez/julius) | MIT | Audio resampling |
| [openunmix](https://github.com/sigsep/open-unmix-pytorch) | MIT | Signal processing utilities |
| [tqdm](https://github.com/tqdm/tqdm) | MPL-2.0 / MIT | Progress bars |
| [PyInstaller](https://www.pyinstaller.org) | GPL-2.0 with exception | Binary packaging (build-time only, exception allows non-free binaries) |

**Note on LGPL:** The compiled binary bundles `libsndfile` as a dynamically-linked shared library (`.dylib`/`.so`/`.dll`), which is compliant with the LGPL-2.1. The library can be replaced by the end user by extracting the binary contents and swapping the shared library file. The full LGPL-2.1 license text is included in the bundled `_soundfile_data/COPYING` inside the binary.
