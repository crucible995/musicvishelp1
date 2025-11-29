#!/usr/bin/env python3
"""
Test script for RunPod VAE endpoint

Usage:
    python test_endpoint.py <audio_file>

Environment variables:
    RUNPOD_ENDPOINT_ID - Your endpoint ID
    RUNPOD_API_KEY - Your API key
"""

import os
import sys
import base64
import requests
import numpy as np

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

def test_endpoint(audio_path: str):
    if not ENDPOINT_ID or not API_KEY:
        print("Error: Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        print("\nExample:")
        print("  export RUNPOD_ENDPOINT_ID=abc123xyz")
        print("  export RUNPOD_API_KEY=your_key_here")
        sys.exit(1)

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"Loading: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode()
    print(f"Audio size: {len(audio_bytes) / 1024 / 1024:.1f} MB")

    print(f"\nCalling RunPod endpoint: {ENDPOINT_ID}")
    print("This may take 30-60 seconds on first call (cold start)...")

    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "audio_base64": audio_b64,
                "compute_umap": True
            }
        },
        timeout=300
    )

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()

    if "error" in result:
        print(f"Error: {result['error']}")
        if "traceback" in result:
            print(result["traceback"])
        sys.exit(1)

    output = result.get("output", result)

    # Decode results
    latents = np.frombuffer(
        base64.b64decode(output["latents"]),
        dtype=np.float32
    ).reshape(output["latents_shape"])

    projection = None
    if "projection" in output:
        projection = np.frombuffer(
            base64.b64decode(output["projection"]),
            dtype=np.float32
        ).reshape(output["projection_shape"])

    print("\n--- Results ---")
    print(f"Duration: {output['duration']:.1f} seconds")
    print(f"Latents: {latents.shape} (num_points x 64)")
    if projection is not None:
        print(f"Projection: {projection.shape} (num_points x 3)")
    print(f"Amplitudes: {len(output['amplitudes'])} values")
    print("\nSuccess! Endpoint is working.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_endpoint.py <audio_file>")
        print("\nExample:")
        print("  python test_endpoint.py song.mp3")
        sys.exit(1)

    test_endpoint(sys.argv[1])
