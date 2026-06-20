#!/usr/bin/env bash
set -e

HOST="100.121.64.70"
USER="ivan"
REMOTE_DIR="~/hubble_pilot"

echo "=== Setting up remote environment ==="

# Create remote directory
ssh "$USER@$HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/output"

# Install Julia (needed for PySR backend)
ssh "$USER@$HOST" << 'REMOTE'
  set -e
  cd ~/hubble_pilot

  # Check if Julia is installed
  if ! command -v julia &> /dev/null; then
    echo "Installing Julia..."
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz
    tar -xf julia-1.10.4-linux-x86_64.tar.gz
    sudo mv julia-1.10.4 /opt/
    sudo ln -sf /opt/julia-1.10.4/bin/julia /usr/local/bin/julia
    rm julia-1.10.4-linux-x86_64.tar.gz
    echo "Julia installed: $(julia --version)"
  else
    echo "Julia already installed: $(julia --version)"
  fi

  # Create Python venv
  if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
  fi

  # Activate and install packages
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel

  # Core ML
  pip install pysr numpy scipy matplotlib pandas

  # Cosmology
  pip install astropy

  # Check if PySR can initialize Julia
  python3 -c "import pysr; pysr.install()" || echo "PySR Julia install may need manual step"

  echo "=== Environment setup complete ==="
REMOTE

echo "Done"
