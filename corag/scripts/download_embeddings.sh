#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../ && pwd )"
echo "working directory: ${DIR}"

E5_INDEX_DIR="${DIR}/data/e5-large-index"
mkdir -p "${E5_INDEX_DIR}"

for i in $(seq 0 39); do
    if [ ! -f "${E5_INDEX_DIR}/e5-large-shard-${i}.pt" ]; then
        echo "Downloading e5-large-shard-${i}.pt"
        wget -q "https://huggingface.co/datasets/corag/kilt-corpus-embeddings/resolve/main/e5-large-shard-${i}.pt" -O "${E5_INDEX_DIR}/e5-large-shard-${i}.pt"
    else
        echo "e5-large-shard-${i}.pt already exists"
    fi
done

echo "Downloading e5-large-index done"
