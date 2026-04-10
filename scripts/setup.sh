#!/bin/bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="grab-rag"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_VERSION="2.6.0"
LLAMA_CPP_VERSION="0.3.16"

[[ ! "$TORCH_CUDA" =~ ^cu[0-9]+$ ]] && { echo "TORCH_CUDA must be a cuda variant like cu124"; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v conda &>/dev/null || { echo "conda not found"; exit 1; }
echo "grab-rag setup, cuda $TORCH_CUDA"

conda env list 2>/dev/null | grep -qw "$ENV_NAME" && echo "$ENV_NAME already exists" || conda env create -f "$REPO_DIR/environment.yaml"
env_path="$(conda env list 2>/dev/null | awk -v name="$ENV_NAME" '$1 == name {print $NF; exit}')"
pip="$env_path/bin/pip"; python="$env_path/bin/python"

echo "installing pytorch $TORCH_VERSION"
"$pip" install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" --force-reinstall
echo "installing llama-cpp-python $LLAMA_CPP_VERSION"
"$pip" install "llama-cpp-python==$LLAMA_CPP_VERSION" --prefer-binary --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$TORCH_CUDA" --force-reinstall
echo "installing requirements"
"$pip" install -r "$REPO_DIR/requirements.txt"

mkdir -p "$REPO_DIR/models" "$REPO_DIR/data" "$REPO_DIR/results"

activate_dir="$env_path/etc/conda/activate.d"; deactivate_dir="$env_path/etc/conda/deactivate.d"
mkdir -p "$activate_dir" "$deactivate_dir"

cat > "$activate_dir/cuda_libs.sh" << 'EOF'
#!/bin/sh
_nvidia_base=""
for _sp in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia; do
    [ -d "$_sp" ] && _nvidia_base="$_sp" && break
done
if [ -n "$_nvidia_base" ]; then
    _cuda_paths=""
    for _d in "$_nvidia_base"/*/lib; do
        [ -d "$_d" ] && _cuda_paths="${_cuda_paths:+$_cuda_paths:}$_d"
    done
    if [ -n "$_cuda_paths" ]; then
        export _GRAB_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$_cuda_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi
EOF

cat > "$deactivate_dir/cuda_libs.sh" << 'EOF'
#!/bin/sh
if [ -n "${_GRAB_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="$_GRAB_OLD_LD_LIBRARY_PATH"
    unset _GRAB_OLD_LD_LIBRARY_PATH
    [ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
fi
EOF

echo "conda activation scripts written"
PYTHONNOUSERSITE=1 "$pip" freeze > "$REPO_DIR/requirements.lock"
conda env export -n "$ENV_NAME" > "$REPO_DIR/environment.lock.yaml"
echo "activate with: conda activate $ENV_NAME"
