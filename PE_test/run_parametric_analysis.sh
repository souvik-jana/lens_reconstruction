#!/bin/bash
# Condor execution script for Parametric Gravitational Lensing PE Analysis

# Get parameters from command line
THETA_E=$1
E1=$2
E2=$3
GAMMA1=$4
GAMMA2=$5

# Set up logging
exec 1> >(tee -a /Users/souvik/Documents/herculens_project/PE_test/logs/parametric_analysis_$(date +%Y%m%d_%H%M%S).log)
exec 2>&1

echo "=========================================="
echo "PARAMETRIC GRAVITATIONAL LENSING PE ANALYSIS"
echo "=========================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Parameters:"
echo "  theta_E = $THETA_E"
echo "  e1 = $E1"
echo "  e2 = $E2"
echo "  gamma1 = $GAMMA1"
echo "  gamma2 = $GAMMA2"
echo ""

# Create necessary directories
mkdir -p /Users/souvik/Documents/herculens_project/PE_test/logs
mkdir -p /Users/souvik/Documents/herculens_project/PE_test/output

# Change to working directory
cd /Users/souvik/Documents/herculens_project/PE_test

# Load conda environment
echo "Loading conda environment..."
source /Users/souvik/miniconda3/etc/profile.d/conda.sh
conda activate herculens-env

# Run the analysis with custom parameters
echo "Starting parametric analysis..."
echo "=========================================="
python herculens_pe_analysis.py --theta_E $THETA_E --e1 $E1 --e2 $E2 --gamma1 $GAMMA1 --gamma2 $GAMMA2

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PARAMETRIC ANALYSIS COMPLETED SUCCESSFULLY"
    echo "End time: $(date)"
    echo "Output files:"
    ls -la output/
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "PARAMETRIC ANALYSIS FAILED"
    echo "End time: $(date)"
    echo "Exit code: $?"
    echo "=========================================="
    exit 1
fi
