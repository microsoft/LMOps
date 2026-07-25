#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-${EL_CONDA_ENV:-el}}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: this setup script currently supports Linux x86_64 only." >&2
    exit 2
fi

if conda env list | awk -v name="${ENV_NAME}" '$1 == name {found=1} END {exit !found}'; then
    echo "Using existing Conda environment: ${ENV_NAME}"
else
    conda create -y -n "${ENV_NAME}" python=3.12
fi

run_in_env() {
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_in_env python -m pip install --no-cache-dir --upgrade \
    pip setuptools==80.10.2 wheel==0.47.0

run_in_env python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

run_in_env python -m pip install --no-cache-dir \
    vllm==0.11.2 transformers==4.57.0

run_in_env python -m pip install --no-cache-dir --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# FlashAttention 2.8.3 needs the pre-4.4 CUTLASS DSL API, while
# FlashInfer requires CUTLASS DSL >=4.2.1.
run_in_env python -m pip uninstall -y \
    nvidia-cutlass-dsl \
    nvidia-cutlass-dsl-libs-base \
    nvidia-cutlass-dsl-libs-core \
    nvidia-cutlass-dsl-libs-cu12
run_in_env python -m pip install --no-cache-dir --no-deps \
    nvidia-cutlass-dsl==4.2.1

run_in_env python -m pip install --no-cache-dir --no-deps \
    tensordict==0.7.1 torchdata==0.11.0

run_in_env python -m pip install --no-cache-dir \
    hydra-core==1.3.2 omegaconf==2.3.0 codetiming==1.4.0 \
    datasets==5.0.0 peft==0.19.1 accelerate==1.14.0 \
    wandb==0.28.1 protobuf==6.33.6 orjson==3.11.9 \
    textarena==0.7.4 trl==0.28.0 rouge-score==0.1.2 \
    evalica==0.3.2 pylatexenc==2.10 pyarrow==25.0.0 pybind11==3.0.4

run_in_env python -m pip install --no-cache-dir --no-deps -e "${REPO_ROOT}/verl"

run_in_env python -c '
import cutlass.cute.core as cutlass_core
import flash_attn
import torch
import transformers
import vllm
from verl.utils.vllm_utils import is_version_ge
from verl.workers.actor.dp_actor import DataParallelPPOActor

assert hasattr(cutlass_core, "ThrMma")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print("EL environment import check passed.")
'

echo "Conda environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
