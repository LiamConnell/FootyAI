# Vertex AI Deployment Guide

Deploy FootyAI training to Google Cloud Vertex AI with automatic cloud storage for videos and model checkpoints.

## Prerequisites

1. **Google Cloud Setup:**
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   
   # Set project
   gcloud config set project learnagentspace
   ```

2. **Docker Setup:**
   ```bash
   # Install Docker if not already installed
   # Enable Container Registry API
   gcloud services enable containerregistry.googleapis.com
   
   # Configure Docker for GCR
   gcloud auth configure-docker
   ```

3. **Enable Required APIs:**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

## Quick Deployment

1. **Update Configuration:**
   Configuration is already set for this project:
   - `PROJECT_ID = "learnagentspace"`
   - `BUCKET_NAME = "footyai"`

2. **Deploy:**
   ```bash
   uv run python deploy_vertex_ai.py
   ```

## What Happens

1. **Docker Image:** Builds container with your training code
2. **Cloud Storage:** Creates bucket for outputs 
3. **Vertex AI Job:** Submits custom training job with T4 GPU
4. **Outputs:**
   - Videos: `gs://footyai/videos/v2_torch_soccer_TIMESTAMP/`
   - Models: `gs://footyai/models/v2_torch_soccer_TIMESTAMP_final.pt`

## Manual Job Submission

If you prefer manual control:

```bash
# Build image
docker build -t gcr.io/learnagentspace/footyai-training .
docker push gcr.io/learnagentspace/footyai-training

# Submit job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=footyai-training-$(date +%Y%m%d-%H%M%S) \
  --config=job_config.yaml
```

## GPU Options

Edit `deploy_vertex_ai.py` to change GPU type:
- `NVIDIA_TESLA_T4` (cheapest, good for testing)
- `NVIDIA_TESLA_V100` (more powerful)
- `NVIDIA_TESLA_A100` (most powerful, expensive)

## Monitoring

- **Console:** https://console.cloud.google.com/vertex-ai/training/custom-jobs
- **Logs:** Click on job → View Logs
- **Storage:** https://console.cloud.google.com/storage/browser

## Cost Estimation

- T4 GPU: ~$0.35/hour
- V100 GPU: ~$2.48/hour  
- A100 GPU: ~$4.89/hour
- Storage: ~$0.02/GB/month

## Troubleshooting

- **Permission errors:** Ensure service account has Vertex AI and Storage permissions
- **Image build fails:** Check Dockerfile and local Docker setup
- **Job fails:** Check logs in Vertex AI console
- **Storage access fails:** Verify bucket exists and permissions are correct