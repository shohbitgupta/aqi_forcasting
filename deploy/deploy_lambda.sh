#!/usr/bin/env bash
# Deploy the prediction service to AWS Lambda from the Docker image, with a public HTTPS link
# (a Lambda Function URL). The model is inside the image, so this deploys both.
#
#   ./deploy/deploy_lambda.sh                          # the first deployment, and every update after it
#   AWS_REGION=eu-west-1 ./deploy/deploy_lambda.sh     # another region (default: ap-south-1, Mumbai)
#
# Needs Docker running and the AWS CLI configured (`aws configure`) with a user allowed to use
# ECR, Lambda and IAM. Safe to run again: it creates what is missing and updates the rest.
# Cost: Lambda's always-free allowance covers this use; the image in ECR costs a few cents a month at most.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
NAME="aqi-forecast"                    # the ECR repository and the Lambda function
ROLE_NAME="aqi-forecast-lambda"        # lets the function write its logs, nothing else
MEMORY_MB=1024
TIMEOUT_S=30                           # the first request after a quiet spell loads the model: a few seconds
MAX_CONCURRENCY=2                      # at most 2 copies at once: a flood of requests slows down, it does not cost more

cd "$(dirname "$0")/.."                # the project root, where the Dockerfile is
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$NAME:$(date +%Y%m%d-%H%M%S)"    # a new tag per deployment, so each one is easy to find
echo "Account $ACCOUNT, region $REGION"

echo "1/5 Image repository (ECR)"
if ! aws ecr describe-repositories --repository-names "$NAME" --region "$REGION" >/dev/null 2>&1; then
    aws ecr create-repository --repository-name "$NAME" --region "$REGION" >/dev/null
fi
# Keep only the 3 newest images, so storage stays at a few cents
aws ecr put-lifecycle-policy --repository-name "$NAME" --region "$REGION" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep the 3 newest images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null

echo "2/5 Build and push the image"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
# arm64: Lambda's cheaper Graviton processors. --provenance=false: Lambda accepts a plain image, not an image index.
docker build --platform linux/arm64 --provenance=false -t "$IMAGE" .
docker push "$IMAGE"

echo "3/5 Permissions (IAM role)"
if ! ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text 2>/dev/null); then
    ROLE_ARN=$(aws iam create-role --role-name "$ROLE_NAME" --query Role.Arn --output text \
        --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}')
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
fi

echo "4/5 Lambda function"
if aws lambda get-function --function-name "$NAME" --region "$REGION" >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$NAME" --region "$REGION" --image-uri "$IMAGE" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"
    aws lambda update-function-configuration --function-name "$NAME" --region "$REGION" \
        --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_S" >/dev/null
    aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"
else
    # A new role takes a few seconds to reach Lambda: retry until it is accepted
    for attempt in 1 2 3 4 5 6; do
        if aws lambda create-function --function-name "$NAME" --region "$REGION" \
            --package-type Image --code ImageUri="$IMAGE" --role "$ROLE_ARN" \
            --architectures arm64 --memory-size "$MEMORY_MB" --timeout "$TIMEOUT_S" >/dev/null; then
            break
        fi
        [ "$attempt" = 6 ] && { echo "Could not create the function"; exit 1; }
        echo "   the role is not ready yet, retrying in 10 s"
        sleep 10
    done
    aws lambda wait function-active-v2 --function-name "$NAME" --region "$REGION"
fi
if ! aws lambda put-function-concurrency --function-name "$NAME" --region "$REGION" \
    --reserved-concurrent-executions "$MAX_CONCURRENCY" >/dev/null 2>&1; then
    # New accounts may have a total limit of 10, and AWS keeps 10 unreserved, so nothing can be reserved
    echo "   note: could not cap concurrency at $MAX_CONCURRENCY (account limit too low); the account limit still caps it"
fi

echo "5/5 Public link (Function URL)"
if ! URL=$(aws lambda get-function-url-config --function-name "$NAME" --region "$REGION" \
        --query FunctionUrl --output text 2>/dev/null); then
    URL=$(aws lambda create-function-url-config --function-name "$NAME" --region "$REGION" \
        --auth-type NONE --query FunctionUrl --output text)
fi
# A public URL needs two permissions (AWS rule since October 2025): calling the URL, and running
# the function through it. Without both, every request gets 403 Forbidden.
POLICY=$(aws lambda get-policy --function-name "$NAME" --region "$REGION" --query Policy --output text 2>/dev/null || true)
if ! grep -q PublicUrlInvokeUrl <<<"$POLICY"; then
    aws lambda add-permission --function-name "$NAME" --region "$REGION" --statement-id PublicUrlInvokeUrl \
        --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
fi
if ! grep -q PublicUrlInvokeFunction <<<"$POLICY"; then
    aws lambda add-permission --function-name "$NAME" --region "$REGION" --statement-id PublicUrlInvokeFunction \
        --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null
fi

URL="${URL%/}"
echo
echo "Deployed: $URL"
echo "Checking $URL/health (the first request loads the model: allow a few seconds)..."
curl -s --max-time 60 "$URL/health" && echo
echo
echo "Test it:   uv run python scripts/test_service.py --url $URL"
echo "Browser:   $URL/docs"
