#!/bin/sh
AWS_REGION="us-east-1"
CONTAINER_TOOL="finch" # Can be docker
LAMBDA_FUNCTION_MEMORY=2048 # MB
LAMBDA_FUNCTION_TIMEOUT=60 # seconds
LAMBDA_FUNCTION_ARCHITECTURE="arm64"
BASE_NAME="yet-another-chatbot"

if ! command -v "${CONTAINER_TOOL}" >/dev/null 2>&1; then
    echo "${CONTAINER_TOOL} does not exist or is not executable. Edit this file to use either 'finch' or 'docker'."
fi

if ! command -v "aws" >/dev/null 2>&1; then
    echo "You need to install the AWS CLI."
fi

IMAGE_NAME="${BASE_NAME}"

echo "Building container image..."
${CONTAINER_TOOL} build --platform linux/arm64 -t ${IMAGE_NAME}:latest .

echo "Using AWS Region: ${AWS_REGION}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Current AWS account ID: ${AWS_ACCOUNT_ID}"

echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | ${CONTAINER_TOOL} login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "Deleting existing ECR repository..."
DELETE_REPO=$(aws ecr delete-repository --region ${AWS_REGION} --repository-name ${IMAGE_NAME} --force)

echo "Creating ECR repository..."
IMAGE_URI=$(aws ecr create-repository --region ${AWS_REGION} --repository-name ${IMAGE_NAME} --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE --query 'repository.repositoryUri' --output text)

IMAGE_URI_WITH_TAG=${IMAGE_URI}:latest

echo "Pushing image to ECR repository..."
${CONTAINER_TOOL} tag ${IMAGE_NAME}:latest ${IMAGE_URI_WITH_TAG}
${CONTAINER_TOOL} push ${IMAGE_URI_WITH_TAG}

echo "Writing image URI to file..."
echo "${IMAGE_URI_WITH_TAG}" > ../Config/image_uri.txt

echo "Image URI: ${IMAGE_URI_WITH_TAG}"

ROLE_NAME="${IMAGE_NAME}-role"

echo "Creating IAM role..."
aws iam create-role --region ${AWS_REGION} --role-name ${ROLE_NAME} --assume-role-policy-document file://trust_policy.json --output text
aws iam attach-role-policy --region ${AWS_REGION} --role-name ${ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

echo "Waiting for IAM role to exist..."
aws iam wait role-exists --region ${AWS_REGION} --role-name ${ROLE_NAME}

ROLE_ARN=$(aws iam get-role --region ${AWS_REGION} --role-name ${ROLE_NAME} --query Role.Arn --output text)
echo "Role ARN: ${ROLE_ARN}"

echo "Waiting for 10 seconds..."
sleep 10

FUNCTION_NAME="${IMAGE_NAME}-function"

echo "Creating or updating Lambda function..."
FUNCTION_ARN=$(aws lambda create-function --region ${AWS_REGION} --function-name ${FUNCTION_NAME} --architectures ${LAMBDA_FUNCTION_ARCHITECTURE} --memory-size ${LAMBDA_FUNCTION_MEMORY} --timeout ${LAMBDA_FUNCTION_TIMEOUT} --package-type Image --code ImageUri=${IMAGE_URI_WITH_TAG} --role ${ROLE_ARN} --query FunctionArn --output text)
echo "Function ARN: ${FUNCTION_ARN}"

echo "Waiting for Lambda function to be active..."
aws lambda wait function-active-v2 --region ${AWS_REGION} --function-name ${FUNCTION_NAME} 

echo "Waiting for Lambda function to be ready for an update..."
aws lambda wait function-updated-v2 --region ${AWS_REGION} --function-name ${FUNCTION_NAME} 

echo "Updating Lambda function configuration..."
FUNCTION_UPDATE_STATUS=$(aws lambda update-function-configuration --region ${AWS_REGION} --function-name ${FUNCTION_NAME} --memory-size ${LAMBDA_FUNCTION_MEMORY} --timeout ${LAMBDA_FUNCTION_TIMEOUT} --query LastUpdateStatus)

echo "Update status: ${FUNCTION_UPDATE_STATUS}"

echo "Waiting for Lambda function update to be completed..."
aws lambda wait function-updated-v2 --region ${AWS_REGION} --function-name ${FUNCTION_NAME} 

echo "Updating Lambda function code..."
FUNCTION_UPDATE_STATUS=$(aws lambda update-function-code --region ${AWS_REGION} --function-name ${FUNCTION_NAME} --architectures ${LAMBDA_FUNCTION_ARCHITECTURE} --image-uri ${IMAGE_URI_WITH_TAG} --query LastUpdateStatus)

echo "Update status: ${FUNCTION_UPDATE_STATUS}"

echo "Waiting for Lambda function update to be completed..."
aws lambda wait function-updated-v2 --region ${AWS_REGION} --function-name ${FUNCTION_NAME} 
