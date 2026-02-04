#!/bin/bash
# =============================================================================
# GNC PROJECT - Environment Setup Script
# =============================================================================
# Creates the Python virtual environment, installs dependencies, sets up
# output directories, and checks for optional toolchain components (MATLAB,
# C++ compiler).
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# =============================================================================

set -e  # Exit on first error

# ---------------------------------------------------------------------------
# Project root (parent of scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "  GNC Project - Environment Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------------------
# 1. Python virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/venv"

if [ -d "$VENV_DIR" ]; then
    echo "[OK] Virtual environment already exists at $VENV_DIR"
else
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created at $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "[OK] Virtual environment activated ($(python3 --version))"
echo ""

# ---------------------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[INFO] Installing Python dependencies..."
pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    pyyaml \
    scikit-learn \
    pytest \
    pytest-cov \
    --quiet

echo "[OK] Python dependencies installed:"
pip list --format=columns | grep -E "numpy|scipy|matplotlib|pandas|PyYAML|scikit-learn|pytest"
echo ""

# ---------------------------------------------------------------------------
# 3. Create output directories
# ---------------------------------------------------------------------------
echo "[INFO] Creating output directories..."

OUTPUT_DIRS=(
    "$PROJECT_ROOT/output/plots"
    "$PROJECT_ROOT/output/trade_studies"
    "$PROJECT_ROOT/output/matlab"
    "$PROJECT_ROOT/output/data"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    mkdir -p "$dir"
    echo "  [OK] $dir"
done
echo ""

# ---------------------------------------------------------------------------
# 4. Check for MATLAB (optional)
# ---------------------------------------------------------------------------
echo "[INFO] Checking for MATLAB..."
if command -v matlab &> /dev/null; then
    MATLAB_VER=$(matlab -batch "disp(version)" 2>/dev/null | tail -1)
    echo "[OK] MATLAB found: $MATLAB_VER"
elif [ -d "/Applications/MATLAB_R"* ] 2>/dev/null; then
    MATLAB_APP=$(ls -d /Applications/MATLAB_R* 2>/dev/null | head -1)
    echo "[OK] MATLAB installation found at: $MATLAB_APP"
    echo "     (Add to PATH if needed: export PATH=\"\$PATH:$MATLAB_APP/bin\")"
else
    echo "[WARNING] MATLAB not found on PATH or in /Applications."
    echo "          MATLAB integration features will be unavailable."
    echo "          The simulation will still run using Python and C++ modules."
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Check for C++ compiler
# ---------------------------------------------------------------------------
echo "[INFO] Checking for C++ compiler..."
CXX_FOUND=false

if command -v g++ &> /dev/null; then
    GXX_VER=$(g++ --version | head -1)
    echo "[OK] g++ found: $GXX_VER"
    CXX_FOUND=true
fi

if command -v clang++ &> /dev/null; then
    CLANG_VER=$(clang++ --version | head -1)
    echo "[OK] clang++ found: $CLANG_VER"
    CXX_FOUND=true
fi

if [ "$CXX_FOUND" = false ]; then
    echo "[WARNING] No C++ compiler (g++ or clang++) found."
    echo "          Install Xcode Command Line Tools (macOS): xcode-select --install"
    echo "          Or install GCC (Linux): sudo apt-get install g++"
    echo "          The C++ real-time simulation module will not be available."
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Check for CMake
# ---------------------------------------------------------------------------
echo "[INFO] Checking for CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version | head -1)
    echo "[OK] CMake found: $CMAKE_VER"
else
    echo "[WARNING] CMake not found. Required to build the C++ module."
    echo "          Install via: brew install cmake (macOS) or apt-get install cmake (Linux)"
fi
echo ""

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  Setup Complete"
echo "=============================================="
echo ""
echo "To activate the virtual environment in a new shell:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the full simulation:"
echo "  ./scripts/run_full_sim.sh"
echo ""
echo "To run tests only:"
echo "  ./scripts/run_tests.sh"
echo ""
