"""
RunPod Serverless Handler for Stable Audio VAE Encoding

This handler accepts audio data and returns latent embeddings + UMAP projection.
Designed for L40 GPU with ~5-20 second cold start when pre-warmed.
"""

import runpod
import torch
import torchaudio
import numpy as np
import json
import base64
import io
import os
import gc

# ============ MODEL LOADING ============

# Global VAE - loaded once at container start
vae = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths (baked into container or mounted)
VAE_CONFIG_PATH = os.environ.get("VAE_CONFIG_PATH", "/models/stable_audio_2_0_vae.json")
VAE_CKPT_PATH = os.environ.get("VAE_CKPT_PATH", "/models/sao_vae_tune_100k_unwrapped.ckpt")

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 2048
LATENT_DIM = 64


def load_vae():
    """Load VAE model at container startup"""
    global vae

    if vae is not None:
        return True

    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

        if not os.path.exists(VAE_CONFIG_PATH):
            raise FileNotFoundError(f"VAE config not found: {VAE_CONFIG_PATH}")
        if not os.path.exists(VAE_CKPT_PATH):
            raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CKPT_PATH}")

        model_config = json.load(open(VAE_CONFIG_PATH))
        vae = create_model_from_config(model_config)
        copy_state_dict(vae, load_ckpt_state_dict(VAE_CKPT_PATH))
        vae.to(device).eval().requires_grad_(False)

        print(f"[RunPod] VAE loaded on {device}")
        print(f"[RunPod] CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return True

    except Exception as e:
        print(f"[RunPod] Failed to load VAE: {e}")
        return False


def encode_audio(waveform: torch.Tensor) -> np.ndarray:
    """Encode audio waveform to latent vectors"""
    global vae

    if vae is None:
        raise RuntimeError("VAE not loaded")

    # Ensure shape: [batch, channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Move to device
    waveform = waveform.to(device)

    with torch.no_grad():
        # Encode through VAE
        latents = vae.encode(waveform)

        # Handle VAE output format
        if hasattr(latents, 'latent_dist'):
            latents = latents.latent_dist.mean
        elif isinstance(latents, tuple):
            latents = latents[0]

        # Shape: [batch, latent_dim, num_latents] -> [num_latents, latent_dim]
        latents = latents.squeeze(0).permute(1, 0).cpu().numpy()

    return latents.astype(np.float32)


def compute_umap_projection(latents: np.ndarray) -> np.ndarray:
    """Project latents to 3D using UMAP"""
    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, len(latents) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )

    projection = reducer.fit_transform(latents)

    # Normalize to roughly [-1, 1] range
    projection = projection - projection.mean(axis=0)
    scale = np.abs(projection).max()
    if scale > 0:
        projection = projection / scale

    return projection.astype(np.float32)


def compute_rms_amplitudes(waveform: torch.Tensor, num_latents: int) -> list:
    """Calculate RMS amplitude for each latent's audio chunk"""
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)

    # Mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() == 2:
        waveform = waveform[0]

    amplitudes = []
    total_samples = waveform.shape[-1]
    samples_per_chunk = total_samples // num_latents

    for i in range(num_latents):
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, total_samples)
        chunk = waveform[start:end]
        rms = torch.sqrt(torch.mean(chunk ** 2)).item()
        amplitudes.append(rms)

    # Normalize
    max_amp = max(amplitudes) if amplitudes else 1.0
    if max_amp > 0:
        amplitudes = [a / max_amp for a in amplitudes]

    return amplitudes


# ============ HANDLER ============

def handler(event):
    """
    RunPod serverless handler

    Input:
        audio_base64: Base64-encoded audio file (WAV, MP3, etc.)
        filename: Original filename (for format detection)
        compute_umap: Whether to compute UMAP projection (default: True)

    Output:
        latents: Base64-encoded float32 numpy array [num_latents, 64]
        projection: Base64-encoded float32 numpy array [num_latents, 3] (if compute_umap)
        amplitudes: List of RMS amplitudes per latent
        num_latents: Number of latent vectors
        duration: Audio duration in seconds
        samples_per_latent: Downsampling ratio (2048)
    """
    try:
        job_input = event.get("input", {})

        # Validate input
        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "Missing audio_base64 in input"}

        compute_umap_flag = job_input.get("compute_umap", True)

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        # Load with torchaudio
        waveform, sr = torchaudio.load(audio_buffer)

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Stereo conversion
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        duration = waveform.shape[1] / SAMPLE_RATE

        # Encode to latents
        print(f"[RunPod] Encoding {duration:.1f}s of audio...")
        latents = encode_audio(waveform)
        num_latents = latents.shape[0]

        # Compute UMAP projection
        projection = None
        if compute_umap_flag and num_latents > 3:
            print(f"[RunPod] Computing UMAP for {num_latents} latents...")
            projection = compute_umap_projection(latents)

        # Compute RMS amplitudes
        amplitudes = compute_rms_amplitudes(waveform, num_latents)

        # Prepare response
        result = {
            "latents": base64.b64encode(latents.tobytes()).decode('ascii'),
            "latents_shape": list(latents.shape),
            "num_latents": num_latents,
            "duration": duration,
            "samples_per_latent": SAMPLES_PER_LATENT,
            "amplitudes": amplitudes,
        }

        if projection is not None:
            result["projection"] = base64.b64encode(projection.tobytes()).decode('ascii')
            result["projection_shape"] = list(projection.shape)

        # Clean up
        del waveform
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[RunPod] Done: {num_latents} latents, {duration:.1f}s")
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ============ STARTUP ============

# Pre-load model at container startup for fast cold starts
print("[RunPod] Initializing Stable Audio VAE worker...")
load_vae()

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
