#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../ && pwd )"
echo "working directory: ${DIR}"

if nc -z localhost 8090; then
  echo "Search server already running."
else
  echo "Starting search server..."

  PYTHONPATH=src/ uvicorn src.search.start_e5_server_main:app --port 8090 > e5_server.log 2>&1 &

  elapsed=0
  while ! nc -z localhost 8090; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ $elapsed -ge 600 ]; then
      echo "Server did not start within 10 minutes. Exiting."
      exit 1
    fi
  done
  echo "Search server started."
fi
