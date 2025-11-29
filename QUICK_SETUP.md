# RunPod Quick Setup Checklist

## Step 1: Add Model Files to runpod/

Copy your VAE model files into this directory:

```bash
cp stable_audio_2_0_vae.json runpod/
cp sao_vae_tune_100k_unwrapped.ckpt runpod/
```

Your runpod/ folder should now contain:
- [ ] `Dockerfile`
- [ ] `rp_handler.py`
- [ ] `requirements.txt`
- [ ] `stable_audio_2_0_vae.json` (you add this)
- [ ] `sao_vae_tune_100k_unwrapped.ckpt` (you add this)

---

## Step 2: Create GitHub Repository

Option A - New repo for just RunPod:
```bash
cd runpod
git init
git add .
git commit -m "RunPod serverless worker"
git remote add origin https://github.com/YOUR_USERNAME/latent-vae-runpod.git
git push -u origin main
```

Option B - Use existing repo with runpod folder:
```bash
git add runpod/
git commit -m "Add RunPod serverless deployment"
git push
```

---

## Step 3: Connect GitHub to RunPod

1. Go to: https://console.runpod.io/user/settings
2. Find **GitHub** under Connections
3. Click **Connect** and authorize RunPod
4. Grant access to your repository

---

## Step 4: Create Serverless Endpoint

1. Go to: https://console.runpod.io/serverless
2. Click **+ New Endpoint**
3. Select **Custom Source** → **GitHub Repository**
4. Choose your repo and branch (usually `main`)
5. Configure:

| Setting | Value |
|---------|-------|
| GPU | L40 (48GB) - recommended |
| Min Workers | 0 |
| Max Workers | 1 |
| Idle Timeout | 30 seconds |
| Execution Timeout | 300 seconds |

6. Click **Deploy**

---

## Step 5: Wait for Build (~15-20 min first time)

RunPod will:
1. Clone your repo
2. Build the Docker image
3. Push to their registry
4. Deploy your endpoint

Watch the build logs for errors.

---

## Step 6: Get Your API Credentials

Once deployed, copy:
- **Endpoint ID**: e.g., `abc123xyz`
- **API Key**: From https://console.runpod.io/user/settings → API Keys

---

## Step 7: Test Your Endpoint

```python
import requests
import base64

# Your credentials
ENDPOINT_ID = "YOUR_ENDPOINT_ID"
API_KEY = "YOUR_API_KEY"

# Load a test audio file
with open("test_song.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Call the endpoint
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={"input": {"audio_base64": audio_b64}},
    timeout=300
)

print(response.json())
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails | Check model files are in runpod/ folder |
| "Model not found" | Verify .ckpt and .json files uploaded |
| Timeout | Increase execution timeout to 600s |
| Out of memory | Use L40 (48GB) instead of smaller GPU |

---

## Costs

- L40: ~$0.76/hour (billed per second)
- Typical encoding: ~30-60 seconds = ~$0.01-0.02 per song
- Monthly (light use): $1-5
