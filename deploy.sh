#!/bin/bash
set -e

PROJECT="forge-steps-ventures"
REGION="us-central1"
APP="bioai-toolkit"
ACCOUNT="mike@stepsventures.com"

export CLOUDSDK_CORE_ACCOUNT="$ACCOUNT"

echo "=== Deploying $APP to Cloud Run ==="
gcloud run deploy "$APP" \
    --source . \
    --region "$REGION" \
    --project "$PROJECT" \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 3

echo ""
echo "=== Deployed ==="
gcloud run services describe "$APP" --region "$REGION" --project "$PROJECT" --format="value(status.url)"
