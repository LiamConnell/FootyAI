#!/usr/bin/env python3
"""
Deploy FootyAI training to Vertex AI Custom Jobs
"""
import subprocess
import os
from datetime import datetime

# Configuration
PROJECT_ID = "learnagentspace"
REGION = "us-central1"  # or your preferred region
BUCKET_NAME = "footyai"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/footyai-training"
JOB_NAME = f"footyai-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def build_and_push_image():
    """Build Docker image and push to Container Registry"""
    print("Building and pushing Docker image...")
    
    # Build the image
    subprocess.run([
        "docker", "build", "-t", IMAGE_URI, "."
    ], check=True)
    
    # Push to GCR
    subprocess.run([
        "docker", "push", IMAGE_URI
    ], check=True)
    
    print(f"Image pushed to {IMAGE_URI}")

def submit_vertex_ai_job():
    """Submit custom job to Vertex AI"""
    print(f"Submitting job {JOB_NAME} to Vertex AI...")
    
    # Submit the job using existing YAML config
    subprocess.run([
        "gcloud", "ai", "custom-jobs", "create",
        "--region", REGION,
        "--display-name", JOB_NAME,
        "--config", "job_config.yaml",
        "--project", PROJECT_ID
    ], check=True)
    
    print(f"Job {JOB_NAME} submitted successfully!")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

def create_bucket_if_not_exists():
    """Create GCS bucket if it doesn't exist"""
    try:
        subprocess.run([
            "gsutil", "ls", f"gs://{BUCKET_NAME}"
        ], check=True, capture_output=True)
        print(f"Bucket gs://{BUCKET_NAME} already exists")
    except subprocess.CalledProcessError:
        print(f"Creating bucket gs://{BUCKET_NAME}...")
        subprocess.run([
            "gsutil", "mb", f"gs://{BUCKET_NAME}"
        ], check=True)

def main():
    print("🚀 Deploying FootyAI to Vertex AI...")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Bucket: gs://{BUCKET_NAME}")
    print(f"Job Name: {JOB_NAME}")
    
    # Check prerequisites
    try:
        subprocess.run(["gcloud", "auth", "list"], check=True, capture_output=True)
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Please ensure gcloud and docker are installed and authenticated")
        return
    
    # Create bucket
    create_bucket_if_not_exists()
    
    # Build and push image
    build_and_push_image()
    
    # Submit job
    submit_vertex_ai_job()
    
    print("✅ Deployment complete!")
    print(f"Videos and models will be saved to gs://{BUCKET_NAME}/")

if __name__ == "__main__":
    main()