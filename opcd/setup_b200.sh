apt-get update && apt-get install -y docker.io wget tmux git curl pip
curl -LsSf https://astral.sh/uv/install.sh | sh ; source $HOME/.local/bin/env
uv venv .venv ; source .venv/bin/activate

uv pip install setuptools
uv pip install -U vllm==0.9.0 --torch-backend=cu128
uv pip install flash-attn --no-build-isolation
uv pip install textarena "trl==0.28.0"
uv pip install "transformers<4.54.0"
uv pip install -U cffi

uv pip install -r requirements_b200.txt

uv pip install -e ./verl

uv pip install datasets==4.0.0
uv pip install rouge-score
uv pip install numba --upgrade

python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install --force-reinstall "setuptools<60"