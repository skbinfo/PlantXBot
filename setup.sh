#!/usr/bin/env bash
# setup.sh
# Robust setup script for PlantXBot

set -euo pipefail

# Configuration
YAML_FILE="requirements.yaml"
TXT_FILE="requirements.txt"
DEFAULT_ENV_NAME="plantxbot-env"

echo "=========================================="
echo "   PlantXBot Environment Setup Script"
echo "=========================================="

# 1. Check for Conda
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' command not found."
  echo "Please install Anaconda or Miniconda first, then restart your terminal."
  exit 1
fi

# Initialize Conda for script usage
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 2. Determine Installation Method
if [[ -f "$YAML_FILE" ]]; then
    echo "Found $YAML_FILE. Using Conda hybrid install (Recommended)."
    
    # Attempt to read environment name from YAML
    ENV_NAME=$(grep "name:" "$YAML_FILE" | head -n1 | awk '{print $2}')
    if [[ -z "$ENV_NAME" ]]; then ENV_NAME="$DEFAULT_ENV_NAME"; fi
    
    echo "Target Environment: $ENV_NAME"

    # Check if env exists
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' already exists."
        read -r -p "Update existing environment? [y/N] " response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Updating environment..."
            conda env update -n "$ENV_NAME" -f "$YAML_FILE" --prune
        else
            echo "Skipping update."
        fi
    else
        echo "Creating new environment '$ENV_NAME'..."
        conda env create -f "$YAML_FILE"
    fi

elif [[ -f "$TXT_FILE" ]]; then
    echo "WARNING: $YAML_FILE not found. Falling back to pip install from $TXT_FILE."
    ENV_NAME="$DEFAULT_ENV_NAME"

    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Creating Python 3.10 environment..."
        conda create -y -n "$ENV_NAME" python=3.10 pip sqlite
    fi
    
    echo "Activating '$ENV_NAME'..."
    conda activate "$ENV_NAME"
    
    echo "Installing dependencies via pip..."
    pip install --upgrade pip setuptools wheel
    pip install -r "$TXT_FILE"
else
    echo "CRITICAL ERROR: Neither $YAML_FILE nor $TXT_FILE found in this directory."
    exit 1
fi

# 3. Post-Install Configuration
echo ""
echo "=== Post-Install Configuration ==="
conda activate "$ENV_NAME"

# Check/Install Spacy Model (Required for NLP tasks)
echo "Checking spaCy model 'en_core_web_sm'..."
if ! python -c "import en_core_web_sm" &> /dev/null; then
    echo "Model not found. Downloading..."
    python -m spacy download en_core_web_sm
else
    echo "spaCy model already installed."
fi

# 4. Final Checks & Instructions
echo ""
echo "=== Setup Verification ==="
if python -c "import flask, chromadb, langchain, torch; print('Core libraries verified.')" 2>/dev/null; then
    echo "SUCCESS: Environment is ready."
else
    echo "WARNING: Some verification checks failed. Review output above for errors."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo ""
echo "To run the app:"
echo "  1. conda activate $ENV_NAME"
echo "  2. python demo_app.py"
echo "=========================================="
