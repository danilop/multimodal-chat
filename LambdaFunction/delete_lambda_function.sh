AWS_REGION="us-west-2"
BASE_NAME="yet-another-chatbot"
ROLE_NAME="${BASE_NAME}-role"
IMAGE_NAME="${BASE_NAME}"
FUNCTION_NAME="${IMAGE_NAME}-function"

echo "Deleting Lambda function..."
aws lambda delete-function --function-name ${FUNCTION_NAME}

echo "Deleting ECR repository..."
aws ecr delete-repository --region ${AWS_REGION} --repository-name ${IMAGE_NAME} --force

echo "Detaching policies from IAM role..."
for POLICY in `aws iam list-attached-role-policies --role-name ${ROLE_NAME} --query 'AttachedPolicies[*].PolicyArn' --output text`; do aws iam detach-role-policy --role-name ${ROLE_NAME} --policy-arn $POLICY; done

echo "Deleting IAM role..."
aws iam delete-role --role-name ${ROLE_NAME}
