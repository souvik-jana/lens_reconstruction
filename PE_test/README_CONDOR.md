# Condor Cluster Setup for Gravitational Lensing PE Analysis

This directory contains all necessary files for running the gravitational lensing parameter estimation analysis on a Condor cluster.

## Files Overview

### Core Analysis
- `herculens_pe_analysis.py` - Main analysis script
- `test_setup.py` - Environment verification script

### Condor Submit Files
- `condor_submit.sub` - Single job submission
- `condor_submit_multiple.sub` - Multiple job submission (5 jobs)
- `condor_submit_parametric.sub` - Parametric sweep submission

### Execution Scripts
- `run_analysis.sh` - Standard analysis execution script
- `run_parametric_analysis.sh` - Parametric analysis execution script

### Monitoring
- `condor_monitor.py` - Job monitoring and management script

## Quick Start

### 1. Single Job Submission
```bash
# Submit a single analysis job
condor_submit condor_submit.sub

# Monitor the job
python condor_monitor.py monitor

# Check job status
python condor_monitor.py status
```

### 2. Multiple Jobs Submission
```bash
# Submit 5 parallel analysis jobs
condor_submit condor_submit_multiple.sub

# Monitor all jobs
python condor_monitor.py monitor
```

### 3. Parametric Sweep
```bash
# Submit parametric sweep across different lens parameters
condor_submit condor_submit_parametric.sub

# Monitor parametric jobs
python condor_monitor.py monitor
```

## Job Configuration

### Resource Requirements
- **Memory**: 8GB per job
- **CPUs**: 4 cores per job
- **Disk**: 2GB per job
- **OS**: Linux x86_64

### File Transfers
- Input files are automatically transferred to worker nodes
- Output files are transferred back upon job completion
- Logs are stored in `logs/` directory

## Monitoring Commands

```bash
# Monitor jobs with 30-second intervals
python condor_monitor.py monitor 30

# Check current queue status
python condor_monitor.py status

# View job history
python condor_monitor.py history

# Check output files
python condor_monitor.py output
```

## Output Structure

```
PE_test/
├── logs/                          # Condor logs
│   ├── <ClusterId>.<Process>.out  # Standard output
│   ├── <ClusterId>.<Process>.err  # Error output
│   └── <ClusterId>.<Process>.log  # Condor log
├── output/                        # Analysis results
│   ├── 01_simulated_observation.png
│   ├── 02_initial_guess.png
│   ├── ...
│   ├── 10_posterior_comparison.png
│   └── samples_all.pkl
└── ...
```

## Customization

### Modify Resource Requirements
Edit the submit files to change:
```bash
RequestMemory = 16GB    # Increase memory
RequestCpus = 8         # Increase CPUs
RequestDisk = 5GB       # Increase disk space
```

### Add More Jobs
Edit `condor_submit_multiple.sub`:
```bash
Queue 10  # Change from 5 to 10 jobs
```

### Custom Parameters
Edit `condor_submit_parametric.sub` to add more parameter combinations:
```bash
# Add new parameter sets
1.0, 0.1, 0.05, 0.01, 0.02
1.2, 0.15, 0.08, 0.015, 0.025
# ... add more lines
```

## Troubleshooting

### Common Issues

1. **Job fails to start**
   - Check if conda environment exists on worker nodes
   - Verify file paths are accessible
   - Check resource requirements

2. **Analysis fails**
   - Check error logs in `logs/` directory
   - Verify all dependencies are installed
   - Check disk space and memory usage

3. **Output files missing**
   - Check if job completed successfully
   - Verify file transfer settings
   - Check disk space on submit node

### Debug Commands

```bash
# Check condor status
condor_status

# View detailed job information
condor_q -long <JobId>

# Check job requirements
condor_q -constraint 'JobId == <JobId>' -long

# View job logs
condor_q -long <JobId> | grep -E "(UserLog|Output|Error)"
```

## Performance Optimization

### For Large Parameter Sweeps
1. Use parametric submission for systematic sweeps
2. Increase memory for complex lens models
3. Use multiple CPUs for parallel processing
4. Consider checkpointing for long-running jobs

### For High Throughput
1. Submit multiple independent jobs
2. Use different random seeds for each job
3. Distribute across multiple cluster nodes
4. Monitor resource usage and adjust accordingly

## Example Workflows

### 1. Quick Test Run
```bash
# Submit single job
condor_submit condor_submit.sub

# Monitor for 5 minutes
python condor_monitor.py monitor 30
```

### 2. Production Run
```bash
# Submit multiple jobs
condor_submit condor_submit_multiple.sub

# Monitor continuously
python condor_monitor.py monitor 60
```

### 3. Parameter Study
```bash
# Submit parametric sweep
condor_submit condor_submit_parametric.sub

# Monitor and collect results
python condor_monitor.py monitor 120
```

## Notes

- All scripts assume conda environment `herculens-env` exists
- Logs are automatically timestamped
- Output files are preserved after job completion
- Jobs can be monitored remotely using the monitoring script
- Failed jobs can be resubmitted using the same submit files
