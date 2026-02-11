#!/bin/bash
# =============================================================================
# GNC PROJECT - Master Simulation Run Script
# =============================================================================
# Executes the complete GNC simulation pipeline:
#   1. Environment verification
#   2. C++ build and benchmark
#   3. Python main simulation
#   4. Trade studies
#   5. Monte Carlo analysis (reduced: 10 runs for quick test)
#   6. Performance benchmarks
#   7. Unit tests
#
# Usage:
#   chmod +x scripts/run_full_sim.sh
#   ./scripts/run_full_sim.sh
# =============================================================================

set -e  # Exit on first error

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_PY="$PROJECT_ROOT/src"
SRC_CPP="$PROJECT_ROOT/src/fsw"
OUTPUT_DIR="$PROJECT_ROOT/output"
BUILD_DIR="$SRC_CPP/build"

# Timestamp for this run
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/data/run_${RUN_TIMESTAMP}.log"

echo "=============================================="
echo "  GNC Project - Full Simulation Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo ""

# Ensure output directories exist
mkdir -p "$OUTPUT_DIR/plots" "$OUTPUT_DIR/trade_studies" "$OUTPUT_DIR/matlab" "$OUTPUT_DIR/data"

# ---------------------------------------------------------------------------
# Step 1: Environment Check
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 1/7: Environment Verification"
echo "----------------------------------------------"

# Check for virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[OK] Virtual environment activated"
else
    echo "[WARNING] Virtual environment not found. Using system Python."
    echo "         Run ./scripts/setup.sh first for a clean environment."
fi

# Verify Python and key packages
python3 -c "import numpy, scipy, pandas, matplotlib, yaml, sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[OK] All Python dependencies available"
else
    echo "[ERROR] Missing Python dependencies. Run ./scripts/setup.sh first."
    exit 1
fi

PYTHONPATH="$SRC_PY:$PYTHONPATH"
export PYTHONPATH
echo "[OK] PYTHONPATH set to include $SRC_PY"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Build C++ Code
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 2/7: C++ Build"
echo "----------------------------------------------"

if command -v cmake &> /dev/null && command -v make &> /dev/null; then
    echo "[INFO] Building C++ real-time simulation module..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) 2>&1 | tail -5
    cd "$PROJECT_ROOT"

    if [ -f "$BUILD_DIR/gnc_rt_sim" ]; then
        echo "[OK] C++ build successful: $BUILD_DIR/gnc_rt_sim"

        # Run C++ benchmark
        echo "[INFO] Running C++ benchmark..."
        "$BUILD_DIR/gnc_rt_sim" > "$OUTPUT_DIR/data/cpp_benchmark_${RUN_TIMESTAMP}.csv" 2>&1 || true
        echo "[OK] C++ benchmark output saved"
    else
        echo "[WARNING] C++ executable not found after build. Check for compile errors."
    fi
else
    echo "[SKIP] CMake or Make not found. Skipping C++ build."
    echo "       Install CMake and a C++ compiler to enable this step."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Python Main Simulation
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 3/7: Main Simulation"
echo "----------------------------------------------"

if [ -f "$SRC_PY/main.py" ]; then
    echo "[INFO] Running main mission simulation..."
    python3 "$SRC_PY/main.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Main simulation complete"
elif [ -f "$SRC_PY/simulation/simulator.py" ]; then
    echo "[INFO] Running simulation module..."
    python3 -m simulation.simulator 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Simulation complete"
else
    echo "[SKIP] No main simulation entry point found (main.py or simulation/simulator.py)."
    echo "       This module may not be implemented yet."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Trade Studies
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 4/7: Trade Studies"
echo "----------------------------------------------"

if [ -f "$SRC_PY/trade_studies/propulsion_trade.py" ]; then
    echo "[INFO] Running propulsion trade study..."
    python3 "$SRC_PY/trade_studies/propulsion_trade.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Propulsion trade study complete"
else
    echo "[SKIP] Trade study module not found."
fi

if [ -f "$SRC_PY/trade_studies/sensor_trade.py" ]; then
    echo "[INFO] Running sensor trade study..."
    python3 "$SRC_PY/trade_studies/sensor_trade.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Sensor trade study complete"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Monte Carlo Analysis (Reduced)
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 5/7: Monte Carlo Analysis (10 runs)"
echo "----------------------------------------------"

if [ -f "$SRC_PY/simulation/monte_carlo.py" ]; then
    echo "[INFO] Running Monte Carlo dispersion analysis (10 runs for quick test)..."
    python3 "$SRC_PY/simulation/monte_carlo.py" --num-runs 10 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Monte Carlo analysis complete"
elif [ -f "$SRC_PY/monte_carlo.py" ]; then
    echo "[INFO] Running Monte Carlo analysis..."
    python3 "$SRC_PY/monte_carlo.py" --num-runs 10 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Monte Carlo analysis complete"
else
    echo "[SKIP] Monte Carlo module not found."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 6: Performance Benchmarks
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 6/7: Performance Benchmarks"
echo "----------------------------------------------"

if [ -f "$SRC_PY/performance/benchmark.py" ]; then
    echo "[INFO] Running performance benchmarks..."
    python3 "$SRC_PY/performance/benchmark.py" 2>&1 | tee -a "$LOG_FILE"
    echo "[OK] Benchmarks complete"
else
    echo "[SKIP] Performance benchmark module not found."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 7: Unit Tests
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Step 7/7: Unit Tests"
echo "----------------------------------------------"

if [ -d "$PROJECT_ROOT/tests" ] || [ -d "$SRC_PY/tests" ]; then
    echo "[INFO] Running pytest..."
    python3 -m pytest "$PROJECT_ROOT" -v --tb=short 2>&1 | tee -a "$LOG_FILE" || true
    echo "[OK] Tests complete"
else
    echo "[SKIP] No test directory found."
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  Pipeline Complete"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Output files generated:"
echo "----------------------------------------------"

# List all output files with sizes
if [ -d "$OUTPUT_DIR" ]; then
    find "$OUTPUT_DIR" -type f -newer "$0" 2>/dev/null | sort | while read -r f; do
        SIZE=$(ls -lh "$f" 2>/dev/null | awk '{print $5}')
        echo "  [$SIZE] $f"
    done
fi

echo ""
echo "Run log: $LOG_FILE"
echo ""
echo "Key output directories:"
echo "  Plots:         $OUTPUT_DIR/plots/"
echo "  Trade studies:  $OUTPUT_DIR/trade_studies/"
echo "  MATLAB export:  $OUTPUT_DIR/matlab/"
echo "  Data/CSV:       $OUTPUT_DIR/data/"
echo ""
