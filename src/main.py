import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.dataset import CHUNK
from src.model import Ludwig
from src.train import get_device

# The model only ever saw 44.1 kHz mono windows while it was training
# so, every output needs to be normalized to that before touching
SAMPLE_RATE = 44100
HOP = CHUNK // 2

def probe_channels(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}: {result.stderr.strip()}")

    output = result.stdout.strip()

    if not output:
        raise SystemExit(f"No audio stream found in {path}")

    return int(output)

def decode_raw(path, channels):
    # decode to interleaved s16le PCM at SAMPLE_RATE
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(channels),
            "-",
        ],
        capture_output=True,
    )

    if result.returncode != 0:
        raise SystemExit(f"ffmpeg decode failed on {path}: {result.stderr.decode().strip()}")

    return result.stdout


def encode_raw(pcm_bytes, channels, out_path):
    # Wrap RAW PCM back into whatever -o extension asks
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-y",
            "-f", "s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(channels),
            "-i", "pipe:0",
            str(out_path),
        ],
        input=pcm_bytes,
        capture_output=True,
    )

    if result.returncode != 0:
        raise SystemExit(f"ffmpeg encode failed on {out_path}: {result.stderr.decode().strip()}")


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train one first with `python -m src.train`."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = Ludwig(channels=checkpoint["channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def overlap_add(model, x, device, batch_frames=64):
    # reconstruct one mono channel with 50% overlap Hann windows

    # Ludwig predicts residual so each window is summed back into his input
    # before the overlap-add

    # Dividing by the accumulated window cancels the fade-in/out at the edges and keeps interior flat
    n = x.shape[0]
    win = torch.hann_window(CHUNK, periodic=True).numpy()

    pad_front = CHUNK - HOP
    padded = np.concatenate(
        [np.zeros(pad_front, dtype=np.float32), x, np.zeros(CHUNK, dtype=np.float32)]
    )

    out = np.zeros(padded.shape[0], dtype=np.float32)
    norm = np.zeros(padded.shape[0], dtype=np.float32)

    frames = list(range(0, padded.shape[0] - CHUNK + 1, HOP))

    for i in range(0, len(frames), batch_frames):
        batch = frames[i:i + batch_frames]
        segs = np.stack([padded[s:s + CHUNK] for s in batch])

        tensor = torch.from_numpy(segs).unsqueeze(1).to(device)

        with torch.no_grad():
            residual = model(tensor).squeeze(1).cpu().numpy()

        recon = segs + residual

        for start, chunk in zip(batch, recon):
            out[start:start + CHUNK] += chunk * win
            norm[start:start + CHUNK] += win

    norm[norm == 0.0] = 1.0

    return (out / norm)[pad_front:pad_front + n]


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a low-passed audio file with Ludwig")

    parser.add_argument("-i", "--input", required=True, help="input audio (any ffmpeg-readable format)")
    parser.add_argument("-o", "--output", required=True, help="reconstructed output (format from extension)")
    parser.add_argument("-m", "--model", default="checkpoint/best.pt", help="checkpoint to use")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: auto)")
    parser.add_argument("--batch-frames", type=int, default=64, help="windows per forward pass")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    model = load_model(args.model, device)

    channels = probe_channels(args.input)
    print(f"Channels: {channels}")

    pcm = decode_raw(args.input, channels)
    samples = len(pcm) // (2 * channels)
    print(f"Samples: {samples} ({samples / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)")

    audio = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).T.astype(np.float32) / 32768.0

    print("Reconstructing...")

    reconstructed = np.empty_like(audio)

    for channel in range(channels):
        reconstructed[channel] = overlap_add(model, audio[channel], device, args.batch_frames)

    reconstructed = np.clip(reconstructed, -1.0, 1.0)

    out_pcm = (reconstructed.T * 32767.0).astype("<i2").tobytes()

    encode_raw(out_pcm, channels, args.output)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
