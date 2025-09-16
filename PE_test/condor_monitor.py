#!/usr/bin/env python3
"""
Condor Job Monitor for Gravitational Lensing PE Analysis
"""

import subprocess
import time
import os
import sys
from datetime import datetime

def run_command(cmd):
    """Run a command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def get_condor_status():
    """Get current condor queue status"""
    stdout, stderr, returncode = run_command("condor_q -submitter")
    return stdout, stderr, returncode

def get_condor_history():
    """Get condor job history"""
    stdout, stderr, returncode = run_command("condor_history -submitter")
    return stdout, stderr, returncode

def monitor_jobs(interval=60):
    """Monitor condor jobs with specified interval"""
    print("="*60)
    print("CONDOR JOB MONITOR - GRAVITATIONAL LENSING PE ANALYSIS")
    print("="*60)
    print(f"Monitoring started at: {datetime.now()}")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("="*60)
    
    try:
        while True:
            print(f"\n[{datetime.now()}] Checking condor queue...")
            
            # Check current queue
            stdout, stderr, returncode = get_condor_status()
            if returncode == 0:
                print("Current queue:")
                print(stdout)
            else:
                print(f"Error checking queue: {stderr}")
            
            # Check if any jobs are running
            if "IDLE" in stdout or "RUNNING" in stdout:
                print("Jobs still running...")
            else:
                print("No jobs in queue. Checking history...")
                hist_stdout, hist_stderr, hist_returncode = get_condor_history()
                if hist_returncode == 0:
                    print("Recent job history:")
                    print(hist_stdout)
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")

def submit_job(submit_file):
    """Submit a condor job"""
    print(f"Submitting job with file: {submit_file}")
    stdout, stderr, returncode = run_command(f"condor_submit {submit_file}")
    
    if returncode == 0:
        print("Job submitted successfully!")
        print(stdout)
        return True
    else:
        print("Error submitting job:")
        print(stderr)
        return False

def check_output_files():
    """Check if output files exist"""
    output_dir = "output"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"Output directory contains {len(files)} files:")
        for file in files:
            filepath = os.path.join(output_dir, file)
            size = os.path.getsize(filepath)
            print(f"  {file} ({size} bytes)")
    else:
        print("Output directory does not exist.")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python condor_monitor.py submit <submit_file>")
        print("  python condor_monitor.py monitor [interval_seconds]")
        print("  python condor_monitor.py status")
        print("  python condor_monitor.py history")
        print("  python condor_monitor.py output")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "submit":
        if len(sys.argv) < 3:
            print("Please specify submit file")
            sys.exit(1)
        submit_file = sys.argv[2]
        submit_job(submit_file)
    
    elif command == "monitor":
        interval = 60
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                print("Invalid interval. Using default 60 seconds.")
        monitor_jobs(interval)
    
    elif command == "status":
        stdout, stderr, returncode = get_condor_status()
        if returncode == 0:
            print(stdout)
        else:
            print(f"Error: {stderr}")
    
    elif command == "history":
        stdout, stderr, returncode = get_condor_history()
        if returncode == 0:
            print(stdout)
        else:
            print(f"Error: {stderr}")
    
    elif command == "output":
        check_output_files()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
