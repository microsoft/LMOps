#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-${L2C_DATA_ROOT:-/tmp/l2c/data}}"

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: Hugging Face CLI not found. Run tools/setup_conda_env.sh first." >&2
    exit 127
fi

mkdir -p "${DATA_DIR}"

for split in train validation test; do
    repo="ytz20/dapo_${split}"
    target="${DATA_DIR}/dapo_${split}.parquet"
    source_path="$(hf download \
        "${repo}" \
        data/train-00000-of-00001.parquet \
        --repo-type dataset)"
    install -m 0644 "${source_path}" "${target}"
done

python - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

data_dir = Path(sys.argv[1])
required = {"content", "data_source", "reward_model"}
for split in ("train", "validation", "test"):
    path = data_dir / f"dapo_{split}.parquet"
    parquet = pq.ParquetFile(path)
    columns = set(pq.read_schema(path).names)
    missing = required - columns
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    if parquet.metadata.num_rows < 1:
        raise SystemExit(f"{path} contains no rows")
    print(f"{path}: {parquet.metadata.num_rows} rows")
PY

echo "All L2C math datasets are available under ${DATA_DIR}"
