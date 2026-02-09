import subprocess
import sys
import os
import time
import atexit

# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the scripts to run in parallel
SCHEDULE_SCRIPTS = [
    "run_scraper_schedule.py",
    "run_processing_schedule.py"
]

processes = []

def cleanup_processes():
    """Ensure all child processes are terminated on exit."""
    print("\nTerminating all child processes...")
    for p in processes:
        if p.poll() is None:  # If the process is still running
            p.terminate()
            p.wait()
    print("All child processes terminated.")

# Register the cleanup function to be called on script exit
atexit.register(cleanup_processes)

def run_in_parallel():
    """Launches multiple scheduler scripts in parallel."""
    python_executable = sys.executable
    
    for script_name in SCHEDULE_SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"--- ERROR: Scheduler script not found at '{script_path}'. Skipping. ---")
            continue

        command = [python_executable, script_path]
        print(f"--- Launching: {' '.join(command)} ---")
        
        process = subprocess.Popen(command, text=True, encoding='utf-8', errors='replace')
        processes.append(process)

    print("\n✅ Both scheduler scripts have been launched in parallel.")
    print("They will run in their own processes.")
    print("Press Ctrl + C in this terminal to stop this main script and all schedulers.")

    try:
        # Keep the main script alive, waiting for the user to interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Ctrl+C detected in main script. Initiating shutdown. ---")
        sys.exit(0)

if __name__ == "__main__":
    run_in_parallel()
