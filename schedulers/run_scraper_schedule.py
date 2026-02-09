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
INITIAL_DEEP_SCRAPE = False
scraper_config = ScraperConfig()

# --- Helper Function to Run Scripts ---
def run_script(script_path, extra_args=None):
    """Executes a Python script as a module to ensure relative imports work."""
    if extra_args is None:
        extra_args = []

    # Convert file path to module path (e.g., core/scrapers/script.py -> FYPII_Project.core.scrapers.script)
    module_path = script_path.replace('/', '.').replace('\\', '.').replace('.py', '')

    python_executable = sys.executable
    command = [python_executable, "-m", module_path] + extra_args
    
    # Set the PYTHONIOENCODING environment variable to ensure UTF-8 output
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    print(f"--- Running command: {' '.join(command)} ---")
    try:
        # Set the working directory to the project root to ensure package discovery
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env, cwd=project_root) as process:
            for line in process.stdout:
                print(line, end='')
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

        print(f"--- SUCCESS: Finished {os.path.basename(script_path)} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Script {os.path.basename(script_path)} failed with exit code {e.returncode}. ---")
        return False
    except FileNotFoundError:
        print(f"--- ERROR: Python executable not found at '{python_executable}'. ---")
        return False

# --- Main Scraper Pipeline Logic ---
def run_scraper_pipeline(deep_scrape=False):
    """Runs the company list download and news scraper."""
    print("🚀 Starting the Scraper Pipeline Run 🚀")

    if not run_script("core/scrapers/company_downloader.py"):
        print("🛑 Scraper run failed: Could not download company list.")
        return

    scraper_args = ["--deep"] if deep_scrape else []
    if not run_script("core/scrapers/news_scraper.py", extra_args=scraper_args):
        print("🛑 Scraper run failed: Could not scrape news.")
        return

    print("\n🎉 --- Scraper Pipeline Run Complete! --- 🎉")

# --- Scheduler Setup ---
if __name__ == "__main__":
    print("📅 Scraper Scheduler starting.")
    print(f"🕒 Performing one initial scraper run (Deep Scrape: {INITIAL_DEEP_SCRAPE})...")
    run_scraper_pipeline(deep_scrape=INITIAL_DEEP_SCRAPE)

    print(f"\n✅ Initial run complete. Scheduling future INCREMENTAL runs every {scraper_config.schedule_minutes} minutes.")
    schedule.every(scraper_config.schedule_minutes).minutes.do(run_scraper_pipeline, deep_scrape=False)

    print("🔁 Running continuously. Press Ctrl + C to stop...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Scraper scheduler stopped manually. Exiting program.")
