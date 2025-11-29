# RunPod Serverless Deployment for Latent Space Explorer

Deploy the Stable Audio VAE encoder as a serverless GPU endpoint on RunPod for faster processing with L40 GPUs.

## Overview

This deployment packages the Stable Audio VAE encoder as a RunPod serverless worker. When connected via GitHub, RunPod automatically:
1. Builds your Docker image from this repository
2. Stores it in RunPod's container registry
3. Deploys it as an auto-scaling serverless endpoint
4. Provides 5-20 second cold starts (model pre-loaded at container startup)

## Model Architecture

Your VAE uses the **Oobleck** encoder/decoder architecture from Stable Audio:

| Parameter | Value |
|-----------|-------|
| Model Type | Autoencoder (VAE) |
| Sample Rate | 44,100 Hz |
| Audio Channels | 2 (stereo) |
| Latent Dimension | 64 |
| Downsampling Ratio | 2048x |
| Encoder Channels | 128 → 256 → 512 → 1024 → 2048 |
| Strides | 2, 4, 4, 8, 8 |

**Key insight**: Every 2048 audio samples becomes 1 latent vector of dimension 64. A 3-minute song at 44.1kHz has ~7.9M samples = ~3,864 latent vectors.

### VRAM Requirements
- **Model weights**: ~500MB - 1GB VRAM
- **Inference buffer**: ~1-2GB for typical songs
- **Recommended GPU**: L40 (48GB) for headroom, also works on A40, RTX 4090, A100

## Files Required

```
runpod/
├── Dockerfile              # Container build instructions
├── rp_handler.py           # RunPod serverless handler
├── requirements.txt        # Python dependencies
├── test_input.json         # Test payload template
└── README.md               # This file

# You must also provide (not in repo for licensing):
/models/
├── stable_audio_2_0_vae.json       # VAE config
└── sao_vae_tune_100k_unwrapped.ckpt # VAE weights (~500MB)
```

## Deployment Steps

### 1. Get Model Weights

You need the Stable Audio VAE checkpoint. Options:

**Option A: Official Stability AI weights** (requires license)
- Contact Stability AI for access

**Option B: Community fine-tunes**
- Search HuggingFace for "stable audio vae"
- Your current weights: `sao_vae_tune_100k_unwrapped.ckpt`

**Option C: Host on cloud storage**
- Upload to S3/GCS/Cloudflare R2
- Modify Dockerfile to download at build time

### 2. Prepare GitHub Repository

Create a new GitHub repo or use a branch with:

```bash
# From your latent-musicvis directory
cd runpod

# Initialize git if needed
git init

# Add files
git add Dockerfile rp_handler.py requirements.txt test_input.json README.md

# Add your model files (see options below)
```

**Model Hosting Options:**

**Option A: Bake into Docker image** (recommended for fast cold starts)
```dockerfile
# Add to Dockerfile before CMD:
COPY stable_audio_2_0_vae.json /models/
COPY sao_vae_tune_100k_unwrapped.ckpt /models/
```
- Pros: Fastest cold start (5-10s), model ready immediately
- Cons: Larger image (~2-3GB), longer build time

**Option B: Download at startup**
```python
# Add to rp_handler.py after imports:
import urllib.request

def download_model():
    if not os.path.exists(VAE_CKPT_PATH):
        print("[RunPod] Downloading model...")
        urllib.request.urlretrieve(
            "https://your-storage.com/sao_vae.ckpt",
            VAE_CKPT_PATH
        )
```
- Pros: Smaller image, easier updates
- Cons: Slower cold start (adds 30-60s for download)

**Option C: Network volume mount** (advanced)
- Create a RunPod network volume with your models
- Mount at `/models` in endpoint config

### 3. Connect GitHub to RunPod

1. Go to [RunPod Console Settings](https://console.runpod.io/user/settings)
2. Under **Connections**, find **GitHub** and click **Connect**
3. Authorize RunPod to access your repository
4. Choose either all repos or specific repos

### 4. Create Serverless Endpoint

1. Go to [RunPod Serverless](https://console.runpod.io/serverless)
2. Click **New Endpoint**
3. Under **Custom Source**, select **GitHub Repository**
4. Select your repository and branch
5. Configure:

| Setting | Recommended Value |
|---------|-------------------|
| GPU Type | L40 (48GB) or A40 (48GB) |
| Min Workers | 0 (scale to zero when idle) |
| Max Workers | 1-3 (based on expected load) |
| Idle Timeout | 30s (keeps container warm) |
| Execution Timeout | 300s (5 min for long files) |

6. Add environment variables if needed:
   - `VAE_CONFIG_PATH`: `/models/stable_audio_2_0_vae.json`
   - `VAE_CKPT_PATH`: `/models/sao_vae_tune_100k_unwrapped.ckpt`

7. Click **Deploy**

### 5. Wait for Build

RunPod will:
1. Clone your repository
2. Build the Docker image (10-20 minutes first time)
3. Push to RunPod's container registry
4. Deploy your endpoint

### 6. Get Your Endpoint URL

Once deployed, you'll get:
- **Endpoint ID**: `abc123xyz`
- **API URL**: `https://api.runpod.ai/v2/abc123xyz/runsync`

## Using the Endpoint

### API Request

```python
import requests
import base64

# Load audio file
with open("song.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Call RunPod endpoint
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "audio_base64": audio_base64,
            "compute_umap": True
        }
    },
    timeout=300
)

result = response.json()

# Decode results
import numpy as np

latents = np.frombuffer(
    base64.b64decode(result["output"]["latents"]),
    dtype=np.float32
).reshape(result["output"]["latents_shape"])

projection = np.frombuffer(
    base64.b64decode(result["output"]["projection"]),
    dtype=np.float32
).reshape(result["output"]["projection_shape"])

print(f"Latents: {latents.shape}")  # (num_latents, 64)
print(f"Projection: {projection.shape}")  # (num_latents, 3)
```

### Async API (for long files)

For files >30 seconds, use async mode:

```python
# Submit job
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={"input": {"audio_base64": audio_base64}}
)
job_id = response.json()["id"]

# Poll for result
while True:
    status = requests.get(
        f"https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/{job_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    ).json()

    if status["status"] == "COMPLETED":
        result = status["output"]
        break
    elif status["status"] == "FAILED":
        raise Exception(status.get("error"))

    time.sleep(1)
```

## Integrating with Latent Space Explorer

Modify your `server.py` to optionally offload encoding to RunPod:

```python
USE_RUNPOD = os.environ.get("USE_RUNPOD", "false").lower() == "true"
RUNPOD_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

async def encode_with_runpod(audio_bytes: bytes):
    """Offload encoding to RunPod GPU"""
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"audio_base64": base64.b64encode(audio_bytes).decode()}}
        )
    return response.json()["output"]
```

## Cold Start Optimization

To achieve 5-20 second cold starts:

1. **Bake models into image** - Avoids download delay
2. **Pre-load in handler** - Model loads before first request
3. **Keep workers warm** - Set idle timeout to 30-60s
4. **Min workers = 1** - Always have one worker ready (costs more)

## Costs

RunPod serverless billing:
- **L40 GPU**: ~$0.76/hr (billed per second of execution)
- **Cold start**: ~$0.01-0.03 per start (5-20 seconds)
- **Encoding**: ~$0.005-0.02 per song (15-60 seconds)

For occasional use (a few songs/day), expect $1-5/month.

## Troubleshooting

### Build Fails
- Check Dockerfile syntax
- Ensure all COPY files exist in repo
- Build must complete in <160 minutes

### Cold Start Too Slow
- Bake models into image
- Check model download speed
- Increase worker idle timeout

### Out of Memory
- Use L40 (48GB) instead of smaller GPU
- Process shorter audio segments
- Enable gradient checkpointing in model

### Model Not Found
- Verify model paths in environment variables
- Check models are in `/models/` directory
- Ensure Dockerfile COPY commands are correct

## Resources

- [RunPod Serverless Docs](https://docs.runpod.io/serverless/get-started)
- [RunPod GitHub Integration](https://docs.runpod.io/serverless/github-integration)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)
