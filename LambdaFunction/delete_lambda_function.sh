AWS_REGION="us-east-1"
BASE_NAME="yet-another-chatbot"
ROLE_NAME="${BASE_NAME}-role"
IMAGE_NAME="${BASE_NAME}"
FUNCTION_NAME="${IMAGE_NAME}-function"

echo "Deleting Lambda function..."
aws lambda delete-function  --region ${AWS_REGION} --function-name ${FUNCTION_NAME}

echo "Deleting ECR repository..."
aws ecr delete-repository --region ${AWS_REGION} --repository-name ${IMAGE_NAME} --force

echo "Detaching policies from IAM role..."
for POLICY in `aws iam list-attached-role-policies  --region ${AWS_REGION} --role-name ${ROLE_NAME} --query 'AttachedPolicies[*].PolicyArn' --output text`; do aws iam detach-role-policy --region ${AWS_REGION} --role-name ${ROLE_NAME} --policy-arn $POLICY; done

echo "Deleting IAM role..."
aws iam delete-role --region ${AWS_REGION} --role-name ${ROLE_NAME}
