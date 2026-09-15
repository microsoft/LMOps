#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-${L2C_CONDA_ENV:-l2c}}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: this setup script supports Linux x86_64 with NVIDIA GPUs." >&2
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
    pip setuptools==80.10.2 wheel==0.45.1

run_in_env python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

run_in_env python -m pip install --no-cache-dir \
    vllm==0.8.5 transformers==4.52.3

run_in_env python -m pip install --no-cache-dir --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

run_in_env python -m pip install --no-cache-dir \
    tensordict==0.6.2 torchdata==0.11.0 \
    ray[default]==2.43.0 \
    hydra-core==1.3.2 omegaconf==2.3.0 codetiming==1.4.0 \
    datasets==3.5.1 peft==0.15.2 accelerate==1.6.0 \
    "numpy<2" "pyarrow>=19,<21" pandas dill packaging \
    pylatexenc pybind11 einops safetensors orjson \
    textarena==0.7.4 math-verify wandb pytest

run_in_env python -m pip install --no-cache-dir --no-deps -e "${REPO_ROOT}/verl"

run_in_env python -c '
import flash_attn
import ray
import textarena
import torch
import transformers
import vllm
from verl.trainer.ppo.l2c_mode import resolve_l2c_mode
from verl.workers.actor.dp_actor import DataParallelPPOActor

mode = resolve_l2c_mode(
    {"l2c_reward_scope": "same_instance", "l2c_num_coaching_rounds": 1},
    "l2l",
)
assert mode.actor_attempts == 2
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__} transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__} ray={ray.__version__}")
print("L2C environment import check passed.")
'

echo "Conda environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
