#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SDCG Production MCMC Run Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script runs the production-level MCMC analysis for SDCG.
#
# Usage:
#   ./run_sdcg_mcmc.sh [OPTIONS]
#
# Options:
#   --test          Run quick test (500 steps, 16 walkers)
#   --quick         Run quick analysis (2000 steps, 32 walkers)
#   --standard      Run standard analysis (10000 steps, 64 walkers)
#   --production    Run full production (50000 steps, 128 walkers) [default]
#   --resume FILE   Resume from checkpoint file
#   --verify-only   Only run unit verification
#
# Requirements:
#   - Python 3.8+
#   - emcee, numpy, scipy
#   - h5py (optional, for HDF5 backend)
#
# Author: CGC Framework
# Date: February 2026
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default settings
RUN_MODE="production"
CHECKPOINT_FILE=""
N_CORES="${N_CORES:-auto}"

# MCMC configurations
declare -A WALKERS=(
    ["test"]=16
    ["quick"]=32
    ["standard"]=64
    ["production"]=128
)

declare -A STEPS=(
    ["test"]=500
    ["quick"]=2000
    ["standard"]=10000
    ["production"]=50000
)

# ─────────────────────────────────────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────────────────────────────────────

VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            RUN_MODE="test"
            shift
            ;;
        --quick)
            RUN_MODE="quick"
            shift
            ;;
        --standard)
            RUN_MODE="standard"
            shift
            ;;
        --production)
            RUN_MODE="production"
            shift
            ;;
        --resume)
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --cores)
            N_CORES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test          Run quick test (500 steps, 16 walkers)"
            echo "  --quick         Run quick analysis (2000 steps, 32 walkers)"
            echo "  --standard      Run standard analysis (10000 steps, 64 walkers)"
            echo "  --production    Run full production (50000 steps, 128 walkers)"
            echo "  --resume FILE   Resume from checkpoint file"
            echo "  --verify-only   Only run unit verification"
            echo "  --cores N       Number of CPU cores to use"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════════════"
echo "SDCG PRODUCTION MCMC"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/cgc_env" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/cgc_env/bin/activate"
elif [ -d "$PROJECT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Verify Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Set working directory
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Unit Verification
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "Running unit verification..."
echo "───────────────────────────────────────────────────────────────────────"

python3 -c "
from cgc.sdcg_units_verification import SDCGUnitVerifier
results = SDCGUnitVerifier.run_comprehensive_verification()
if not results['all_passed']:
    import sys
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Unit verification FAILED"
    echo "  Please fix the issues before running MCMC"
    exit 1
fi

if [ "$VERIFY_ONLY" = true ]; then
    echo ""
    echo "✓ Unit verification complete (--verify-only mode)"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# MCMC Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_WALKERS=${WALKERS[$RUN_MODE]}
N_STEPS=${STEPS[$RUN_MODE]}

echo ""
echo "MCMC Configuration:"
echo "───────────────────────────────────────────────────────────────────────"
echo "  Mode:           $RUN_MODE"
echo "  Walkers:        $N_WALKERS"
echo "  Steps:          $N_STEPS"
echo "  Checkpoint:     ${CHECKPOINT_FILE:-none}"
echo "  CPU cores:      $N_CORES"
echo ""

# Estimate runtime
if [ "$RUN_MODE" = "production" ]; then
    echo "  Estimated time: 24-48 hours (local) / 4-8 hours (cluster)"
elif [ "$RUN_MODE" = "standard" ]; then
    echo "  Estimated time: 4-8 hours (local) / 1-2 hours (cluster)"
elif [ "$RUN_MODE" = "quick" ]; then
    echo "  Estimated time: 30-60 minutes"
else
    echo "  Estimated time: 5-10 minutes"
fi

echo ""

# Confirm for production runs
if [ "$RUN_MODE" = "production" ]; then
    read -p "Start production run? This may take 24+ hours. [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run MCMC
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting MCMC..."
echo "═══════════════════════════════════════════════════════════════════════"

START_TIME=$(date +%s)

# Build command
CMD="python3 -m cgc.sdcg_production_mcmc"
CMD="$CMD --n-walkers $N_WALKERS"
CMD="$CMD --n-steps $N_STEPS"

if [ -n "$CHECKPOINT_FILE" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT_FILE"
fi

# Run with optional parallelization
if command -v mpirun &> /dev/null && [ "$N_CORES" != "1" ]; then
    # MPI available - use for production runs
    if [ "$RUN_MODE" = "production" ] || [ "$RUN_MODE" = "standard" ]; then
        if [ "$N_CORES" = "auto" ]; then
            N_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        fi
        echo "Using MPI with $N_CORES processes..."
        mpirun -n "$N_CORES" $CMD
    else
        $CMD
    fi
else
    $CMD
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "MCMC RUN COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Mode:           $RUN_MODE"
echo "  Elapsed time:   $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "  Results in:     $PROJECT_DIR/results/"
echo ""

# Find latest results directory
LATEST_RESULTS=$(ls -td "$PROJECT_DIR/results/sdcg_mcmc_"* 2>/dev/null | head -1)

if [ -n "$LATEST_RESULTS" ]; then
    echo "  Latest results: $LATEST_RESULTS"
    
    if [ -f "$LATEST_RESULTS/final_summary.txt" ]; then
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "PARAMETER CONSTRAINTS:"
        echo "───────────────────────────────────────────────────────────────────────"
        grep -A 20 "PARAMETER CONSTRAINTS" "$LATEST_RESULTS/final_summary.txt" | head -15
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
