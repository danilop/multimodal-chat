#!/bin/bash

echo "Enter your password:"
read -s PASSWD

echo "OPENSEARCH_INITIAL_ADMIN_PASSWORD='${PASSWD}'" > .env
echo "export OPENSEARCH_PASSWORD='${PASSWD}'" > opensearch_env.sh  

echo "Password written to .env (for docker-compose.yml) and opensearch_env.sh (for multimodal_cat.py)."
export OPENSEARCH_PASSWORD='@PasSw0rd57.3'
