#!/bin/bash
# Condor execution script for Gravitational Lensing PE Analysis

# Set up logging
exec 1> >(tee -a /Users/souvik/Documents/herculens_project/PE_test/logs/analysis_$(date +%Y%m%d_%H%M%S).log)
exec 2>&1

echo "=========================================="
echo "GRAVITATIONAL LENSING PE ANALYSIS - CONDOR"
echo "=========================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
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

# Verify environment
echo "Verifying environment..."
python test_setup.py

# Run the analysis
echo ""
echo "Starting analysis..."
echo "=========================================="
python herculens_pe_analysis.py

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ANALYSIS COMPLETED SUCCESSFULLY"
    echo "End time: $(date)"
    echo "Output files:"
    ls -la output/
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ANALYSIS FAILED"
    echo "End time: $(date)"
    echo "Exit code: $?"
    echo "=========================================="
    exit 1
fi
