import subprocess
import time
import schedule
import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import ScraperConfig

# --- Configuration ---
scraper_config = ScraperConfig()

# --- Helper Function to Run Scripts ---
def run_script(module_name, extra_args=None):
    """Executes a Python module using the -m flag."""
    if extra_args is None:
        extra_args = []

    python_executable = sys.executable
    command = [python_executable, "-m", module_name] + extra_args
    
    print(f"--- Running command: {' '.join(command)} ---")
    try:
        with subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            cwd=project_root
        ) as process:
            for line in process.stdout:
                print(line, end='')
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

        print(f"--- SUCCESS: Finished {module_name} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Module {module_name} failed with exit code {e.returncode}. ---")
        return False
    except FileNotFoundError:
        print(f"--- ERROR: Python executable not found at '{python_executable}'. ---")
        return False

# --- Main Processing Pipeline Logic ---
def run_processing_pipeline():
    """Runs the post-scraping processing scripts."""
    print("⚙️ Starting the Post-Scraping Processing Run ⚙️")

    if not run_script("core.processors.company_extractor"):
        print("🛑 Processing run failed: Could not extract companies.")
        return

    if not run_script("core.processors.embedding_processor"):
        print("🛑 Processing run failed: Could not embed chunks.")
        return

    print("\n✅ --- Processing Run Complete! --- ✅")

# --- Scheduler Setup ---
if __name__ == "__main__":
    print("📅 Post-Scraping Processing Scheduler starting.")
    print(f"🕒 Scheduling runs every {scraper_config.schedule_minutes} minute(s).")
    
    run_processing_pipeline()  # Run once immediately at the start
    schedule.every(scraper_config.schedule_minutes).minutes.do(run_processing_pipeline)

    print("🔁 Running continuously. Press Ctrl + C to stop...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Processing scheduler stopped manually. Exiting program.")