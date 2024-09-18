CONTAINER_TOOL="finch" # Can be docker
COMPOSE_TOOL="finch compose" # Can be docker-compose
source opensearch_env.sh
${COMPOSE_TOOL} pull
${CONTAINER_TOOL} image prune -f