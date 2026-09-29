#!/usr/bin/env bash
#
# Build a working environment for the mamba_attn_hybrid draft model.
#
# Clones the two pinned upstream repositories, applies the patches that sit next
# to this script, and installs everything into a virtualenv.
#
# Usage:
#   ./apply.sh [TARGET_DIR]        # default: the directory holding this script
#
# Environment:
#   SKIP_MAMBA=1   skip causal-conv1d and mamba_ssm (needed for training only,
#                  and they require a CUDA toolkit to compile)
#   PYTHON=3.13    Python version for the virtualenv

set -euo pipefail

SPECULATORS_URL=https://github.com/vllm-project/speculators
SPECULATORS_REF=3251ed0
VLLM_URL=https://github.com/vllm-project/vllm
VLLM_REF=v0.25.1

# The precompiled wheel must come from the same commit as the source tree being
# patched, or the C extensions and the Python code disagree. This is v0.25.1.
VLLM_WHEEL_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${1:-$HERE}
PYTHON=${PYTHON:-3.13}
PY=$ROOT/.venv/bin/python

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

for tool in git uv; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is not installed"
done
for p in speculators.patch vllm.patch; do
    [ -f "$HERE/$p" ] || die "$HERE/$p not found (run this script from the repository)"
done

# Clone at a pinned ref. Skips work if the directory is already at that ref.
clone_at() {
    local url=$1 dir=$2 ref=$3
    if [ -d "$dir/.git" ]; then
        say "$dir already exists, checking out $ref"
    else
        say "cloning $url at $ref"
        git clone --filter=blob:none "$url" "$dir"
    fi
    git -C "$dir" fetch --tags origin "$ref" 2>/dev/null || true
    git -C "$dir" checkout --detach "$ref"
}

# Apply a patch, tolerating the case where it is already applied.
apply_patch() {
    local dir=$1 patch=$2
    if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        say "$(basename "$patch") is already applied, skipping"
        return 0
    fi
    git -C "$dir" apply --check "$patch" \
        || die "$(basename "$patch") does not apply to $dir at the pinned ref"
    git -C "$dir" apply "$patch"
    say "$(basename "$patch") applied to $dir"
}

mkdir -p "$ROOT"

clone_at "$SPECULATORS_URL" "$ROOT/speculators" "$SPECULATORS_REF"
apply_patch "$ROOT/speculators" "$HERE/speculators.patch"

clone_at "$VLLM_URL" "$ROOT/vllm" "$VLLM_REF"
apply_patch "$ROOT/vllm" "$HERE/vllm.patch"

# Guard the wheel pin against the checked-out source.
actual=$(git -C "$ROOT/vllm" rev-parse HEAD)
[ "$actual" = "$VLLM_WHEEL_COMMIT" ] \
    || die "vllm is at $actual but the wheel pin is $VLLM_WHEEL_COMMIT; these must match"


say "creating virtualenv at $ROOT/.venv (python $PYTHON)"
uv venv --python "$PYTHON" "$ROOT/.venv"

say "installing released vLLM to pull in torch and the rest of the dependency tree"
uv pip install --python "$PY" "vllm==0.25.1"

say "installing build dependencies"
uv pip install --python "$PY" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 ninja

# Replaces the released vLLM with the patched copy. The patch touches Python
# files only, so this reuses the released binaries instead of rebuilding them.
say "installing the patched vLLM over it"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_WHEEL_COMMIT" \
    uv pip install --python "$PY" --no-build-isolation --no-deps "$ROOT/vllm"

say "installing the patched speculators (editable)"
uv pip install --python "$PY" -e "$ROOT/speculators"

if [ "${SKIP_MAMBA:-0}" = "1" ]; then
    say "skipping causal-conv1d and mamba_ssm (SKIP_MAMBA=1); serving does not need them"
else
    say "installing causal-conv1d and mamba_ssm (training kernels; this takes a while)"
    uv pip install --python "$PY" --no-build-isolation causal-conv1d mamba_ssm
fi

say "verifying the patched modules import"
"$PY" -c "from vllm.model_executor.models import qwen3_dflash_mamba_attn_hybrid" \
    || die "the vLLM patch did not take effect"
"$PY" -c "import speculators.models.mamba_attn_hybrid" \
    || die "the speculators patch did not take effect"

say "done. activate with: source $ROOT/.venv/bin/activate"
