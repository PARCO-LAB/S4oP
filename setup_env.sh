#!/bin/bash
set -euo pipefail

# ----------------------------- parametri ------------------------------------
PROJECT_DIR="/storage-large/homedirs/dnemrc43/S4oP"
VENV_NAME="venv313"
PY_VERSION="3.13"
CUDA_MODULE="cuda/12.8"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
MAMBA_VERSION="2.3.0"
export MAX_JOBS=2

cd "$PROJECT_DIR"

# --------- abilita il comando 'module' anche in shell non interattiva -------
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi

# ----------------------------- 1. uv ----------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ----------------- 2. Python 3.13 (gestito da uv, con header) ---------------
uv python install "$PY_VERSION"

# ----------------------------- 3. venv --------------------------------------
uv venv "$VENV_NAME" --python "$PY_VERSION" --python-preference only-managed
source "$VENV_NAME/bin/activate"

# ----------------------- 4. controllo header --------------------------------
python -c "import sysconfig, os; p=sysconfig.get_path('include'); assert os.path.exists(os.path.join(p,'Python.h')), 'Python.h MANCANTE'; print('header OK:', p)"

# --------------------------- 5. CUDA toolkit --------------------------------
module load "$CUDA_MODULE"
nvcc --version

# ------------- 6. torch (pinnato) + requirements base S4oP ------------------
# torch/torchvision pinnati alla combo validata; mamba-ssm/causal-conv1d/pykeops
# vengono esclusi dal requirements e gestiti a parte sotto.
uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
grep -vE '^[[:space:]]*(torchvision|torch|mamba-ssm|causal-conv1d|pykeops)([[:space:]=<>!~]|$)' requirements.txt > /tmp/req_base.txt
uv pip install -r /tmp/req_base.txt
uv pip install packaging ninja

# ------------ 7. kernel Mamba (compilano da sorgente, ~15-20 min) -----------
uv pip install "mamba-ssm==$MAMBA_VERSION" --no-build-isolation
uv pip install causal-conv1d --no-build-isolation

# ----------------------------- 8. verifica ----------------------------------
python - <<'PYEOF'
import torch
from causal_conv1d import causal_conv1d_fn
from mamba_ssm import Mamba
m = Mamba(d_model=64).cuda()
y = m(torch.randn(2, 16, 64, device="cuda"))
print("causal_conv1d_fn:", causal_conv1d_fn is not None)
print("Mamba output  :", tuple(y.shape))
print("=== SETUP OK ===")
PYEOF

echo ""
echo "Ambiente pronto in: $PROJECT_DIR/$VENV_NAME"
echo "Per usarlo nei job:  source $VENV_NAME/bin/activate && module load $CUDA_MODULE"