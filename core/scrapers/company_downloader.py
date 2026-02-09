import os
import json
import time
import pypdf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import sys
import os

# Add project root to Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DataConfig
def download_and_extract_listed_companies():
    """
    Downloads the PDF of listed companies from Bursa Malaysia using Selenium,
    extracts the company names and stock codes, and saves them to a JSON file.
    """
    data_config = DataConfig()
    pdf_url = "https://www.bursamalaysia.com/sites/5d809dcf39fba22790cad230/assets/66a71153e6414a8b25f23ecc/List_of_Companies.pdf"
    script_dir = os.path.abspath(os.path.dirname(__file__))
    pdf_filename = "List_of_Companies.pdf"
    download_path = os.path.join(script_dir, pdf_filename)
    json_path = data_config.listed_companies_json_path
    driver = None

    try:
        # --- Download the PDF using Selenium ---
        print("Setting up Chrome for PDF download...")
        chrome_options = Options()
        prefs = {
            "download.default_directory": script_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,  # Important for downloading
        }
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")

        print("Initializing WebDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"Navigating to {pdf_url} to start download...")
        driver.get(pdf_url)

        # --- Wait for the download to complete ---
        timeout = 60  # seconds
        start_time = time.time()
        print(f"Waiting for download to complete at: {download_path}")

        while not os.path.exists(download_path):
            if time.time() - start_time > timeout:
                raise TimeoutError("PDF download timed out.")
            time.sleep(1)

        print(f"Successfully downloaded {pdf_filename}")

        # --- Extract data from the downloaded PDF ---
        print(f"Extracting text from {download_path}...")
        companies = []
        with open(download_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            for page_num in range(1, len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 3 and parts[0].isdigit() and parts[-1].isdigit():
                        stock_code = parts[-2]
                        if len(stock_code) >= 4:
                            company_name = " ".join(parts[1:-2])
                            companies.append({
                                "company": company_name,
                                "ticker": stock_code
                            })
        
        # --- Save the extracted data to JSON ---
        if not companies:
            print("No companies were extracted. The JSON file will not be created.")
            return
        
        print(f"Found {len(companies)} companies.")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(companies, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved data to {json_path}")

    except Exception as e:
        error_message = f"An error occurred during the process: {e}"
        print(error_message.encode('utf-8', errors='replace').decode('utf-8'))
    finally:
        # --- Comprehensive Cleanup ---
        if driver:
            driver.quit()
            print("WebDriver quit.")

        # Clean up the downloaded PDF file
        if os.path.exists(download_path):
            try:
                os.remove(download_path)
                print(f"Removed temporary file: {download_path}")
            except Exception as e:
                print(f"Error removing PDF file {download_path}: {e}")

        # Clean up temporary .tmp files
        print("Cleaning up temporary .tmp files...")
        for item in os.listdir(script_dir):
            if item.endswith(".tmp"):
                item_path = os.path.join(script_dir, item)
                try:
                    os.remove(item_path)
                    print(f"Removed temporary file: {item}")
                except Exception as e:
                    print(f"Error removing temporary file {item}: {e}")

if __name__ == "__main__":
    download_and_extract_listed_companies()