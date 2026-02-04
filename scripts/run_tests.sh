#!/bin/bash
# =============================================================================
# GNC PROJECT - Test Runner
# =============================================================================
# Runs the full test suite for both Python and C++ components with verbose
# output and coverage reporting.
#
# Usage:
#   chmod +x scripts/run_tests.sh
#   ./scripts/run_tests.sh
# =============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_PY="$PROJECT_ROOT/src/python"
SRC_CPP="$PROJECT_ROOT/src/cpp"
BUILD_DIR="$SRC_CPP/build"

PYTHON_PASS=0
PYTHON_FAIL=0
CPP_PASS=0
CPP_FAIL=0

echo "=============================================="
echo "  GNC Project - Test Suite"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[OK] Virtual environment activated"
else
    echo "[INFO] No virtual environment found. Using system Python."
fi

PYTHONPATH="$SRC_PY:$PYTHONPATH"
export PYTHONPATH
echo ""

# ---------------------------------------------------------------------------
# Python Tests (pytest with coverage)
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  Python Tests (pytest)"
echo "----------------------------------------------"

# Find test directories
TEST_DIRS=""
if [ -d "$PROJECT_ROOT/tests" ]; then
    TEST_DIRS="$PROJECT_ROOT/tests"
fi
if [ -d "$SRC_PY/tests" ]; then
    TEST_DIRS="$TEST_DIRS $SRC_PY/tests"
fi

if [ -n "$TEST_DIRS" ]; then
    echo "[INFO] Running pytest with coverage..."
    echo ""

    python3 -m pytest $TEST_DIRS \
        -v \
        --tb=short \
        --cov="$SRC_PY" \
        --cov-report=term-missing \
        --cov-report=html:"$PROJECT_ROOT/output/data/htmlcov" \
        -x \
        2>&1

    PYTEST_EXIT=$?

    if [ $PYTEST_EXIT -eq 0 ]; then
        echo ""
        echo "[PASS] All Python tests passed."
        PYTHON_PASS=1
    elif [ $PYTEST_EXIT -eq 5 ]; then
        echo ""
        echo "[INFO] No Python tests collected. Test files may not exist yet."
        PYTHON_PASS=1
    else
        echo ""
        echo "[FAIL] Some Python tests failed (exit code: $PYTEST_EXIT)."
        PYTHON_FAIL=1
    fi
else
    echo "[SKIP] No test directories found (tests/ or src/python/tests/)."
    echo "       Create test files following the pattern: test_*.py"
fi
echo ""

# ---------------------------------------------------------------------------
# C++ Tests
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo "  C++ Tests"
echo "----------------------------------------------"

CPP_TEST_BIN="$BUILD_DIR/gnc_rt_test"

if [ -f "$CPP_TEST_BIN" ]; then
    echo "[INFO] Running C++ test executable..."
    echo ""

    "$CPP_TEST_BIN" 2>&1
    CPP_EXIT=$?

    if [ $CPP_EXIT -eq 0 ]; then
        echo ""
        echo "[PASS] All C++ tests passed."
        CPP_PASS=1
    else
        echo ""
        echo "[FAIL] C++ tests failed (exit code: $CPP_EXIT)."
        CPP_FAIL=1
    fi
elif [ -d "$BUILD_DIR" ]; then
    # Try to find any test executables in the build directory
    FOUND_TESTS=$(find "$BUILD_DIR" -name "*test*" -perm +111 -type f 2>/dev/null)
    if [ -n "$FOUND_TESTS" ]; then
        for test_bin in $FOUND_TESTS; do
            echo "[INFO] Running: $test_bin"
            "$test_bin" 2>&1
            if [ $? -eq 0 ]; then
                echo "[PASS] $test_bin"
                CPP_PASS=1
            else
                echo "[FAIL] $test_bin"
                CPP_FAIL=1
            fi
        done
    else
        echo "[SKIP] No C++ test executables found in $BUILD_DIR."
        echo "       Build with: cd src/cpp/build && cmake .. && make"
    fi
else
    echo "[SKIP] C++ not built. Run ./scripts/run_full_sim.sh to build first."
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  Test Summary"
echo "=============================================="

TOTAL_PASS=$((PYTHON_PASS + CPP_PASS))
TOTAL_FAIL=$((PYTHON_FAIL + CPP_FAIL))

echo ""
echo "  Python tests:  $([ $PYTHON_PASS -eq 1 ] && echo 'PASS' || ([ $PYTHON_FAIL -eq 1 ] && echo 'FAIL' || echo 'SKIP'))"
echo "  C++ tests:     $([ $CPP_PASS -eq 1 ] && echo 'PASS' || ([ $CPP_FAIL -eq 1 ] && echo 'FAIL' || echo 'SKIP'))"
echo ""

if [ -d "$PROJECT_ROOT/output/data/htmlcov" ]; then
    echo "  Coverage report: $PROJECT_ROOT/output/data/htmlcov/index.html"
    echo ""
fi

echo "End time: $(date)"
echo ""

# Exit with failure code if any tests failed
if [ $TOTAL_FAIL -gt 0 ]; then
    echo "[RESULT] SOME TESTS FAILED"
    exit 1
else
    echo "[RESULT] ALL TESTS PASSED (or skipped)"
    exit 0
fi
