#!/bin/bash
PING_URL="https://Fieerawe-cybersec-soc-env.hf.space"
REPO_DIR="/c/Users/HP/Documents/openenv/cybersec_soc_env"

echo "========================================"
echo "OpenEnv Submission Validator"
echo "========================================"

echo "Step 1: HF Space ping..."
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30)
[ "$CODE" = "200" ] && echo "PASSED -- HF Space live" || echo "FAILED -- Got $CODE"

echo "Step 2: Docker build..."
if [ -f "$REPO_DIR/Dockerfile" ]; then
    docker build "$REPO_DIR" > /dev/null 2>&1 && echo "PASSED -- Docker build succeeded" || echo "FAILED"
else
    echo "FAILED -- No Dockerfile found"
fi

echo "Step 3: openenv validate..."
cd "$REPO_DIR"
openenv validate && echo "PASSED" || echo "FAILED"

echo "========================================"
echo "Done!"
