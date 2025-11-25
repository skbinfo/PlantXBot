#!/usr/bin/env bash
# setup_full.sh
# Create a conda env 'chatbot_env' and install exact packages from requirements.txt
# Intended to reproduce the environment you showed exactly.
set -euo pipefail

ENV_NAME="chatbot_env"
PYTHON_VERSION="3.10"
REQ_FILE="requirements.txt"
FROZEN_OUT="installed_requirements.txt"
ENV_YML="environment.yml"

echo "== Setup full reproducible environment: $ENV_NAME =="

# Check for conda
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found in PATH. Install Miniconda/Anaconda first and re-run."
  exit 1
fi

# Ensure conda shell functions are available
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
  read -r -p "Do you want to remove and recreate it (y/N)? " yn
  case "$yn" in
    [Yy]* )
      echo "Removing existing environment '$ENV_NAME'..."
      conda env remove -y -n "$ENV_NAME"
      ;;
    * )
      echo "Leaving existing environment; activating it."
      ;;
  esac
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda environment '$ENV_NAME' with python $PYTHON_VERSION..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Upgrading pip, wheel, setuptools..."
python -m pip install --upgrade pip wheel setuptools

if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: $REQ_FILE not found in $(pwd). Please put the exact requirements.txt in the same folder as this script."
  exit 1
fi

echo
echo "== Installing packages from $REQ_FILE via pip =="
echo "(This may take a long time and produce a lot of output.)"
python -m pip install -r "$REQ_FILE"

echo
echo "== Post-install: installing spaCy model en_core_web_sm if needed =="
python - <<'PY'
import importlib, subprocess, sys
try:
    import en_core_web_sm  # installed as package
    print("spaCy model 'en_core_web_sm' already importable.")
except Exception:
    print("Attempting to download 'en_core_web_sm' via spacy cli...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except Exception as e:
        print("Warning: failed automatic spaCy model download. You can run:")
        print("  python -m spacy download en_core_web_sm")
PY

echo
echo "== Exporting conda environment.yml (no-builds) =="
conda env export -n "$ENV_NAME" --no-builds > "$ENV_YML" || echo "Warning: conda env export failed."

echo
echo "== Capturing pip freeze to $FROZEN_OUT =="
python -m pip freeze > "$FROZEN_OUT"

echo
echo "== Comparison summary (requirements.txt vs installed pip freeze) =="
if command -v diff >/dev/null 2>&1; then
  echo "Showing first 200 lines of 'diff -u requirements.txt installed_requirements.txt' (if any):"
  diff -u --label "requirements.txt" --label "$FROZEN_OUT" "$REQ_FILE" "$FROZEN_OUT" | sed -n '1,200p' || true
  echo
  echo "If diff reported differences, inspect $FROZEN_OUT and $ENV_YML to reconcile."
else
  echo "Note: 'diff' command not available, skipping textual diff. You can run:"
  echo "  diff -u requirements.txt $FROZEN_OUT"
fi

echo
echo "== NVIDIA / CUDA note =="
echo "Your original environment included many 'nvidia-cuda-*' and 'nvidia-*' packages."
cat <<'WARN'
These packages often require system-level CUDA drivers and/or conda-managed CUDA toolkits.
If you plan to use GPU acceleration, prefer installing pytorch / cuda packages via conda from the pytorch and nvidia channels:

Example (edit cuda version to match your drivers):
  conda install -y -n chatbot_env -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.4

If you already have working driver support and want the exact pip-installed nvidia packages, the pip install step above attempted to install them, but pip-installed nvidia packages may be incomplete without matching system libraries.
WARN

echo
echo "== Completed =="
echo "Files created:"
echo "  - $ENV_YML"
echo "  - $FROZEN_OUT"
echo
echo "Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo
echo "Run a quick check, for example:"
echo "  python -c \"import flask,langchain,chromadb,torch; print('OK')\""
echo
echo "If you want, keep these files in the repo and tag commit with the exact git tag for reproducibility (git tag -a v1.0 -m 'repro v1.0')."

