import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json

# Set up Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')  # Enable headless mode
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--window-size=1920,1080')  # Set screen size
    # chrome_options.add_argument('--headless')  # Uncomment for headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver
# Save data to a JSONL file
def save_to_jsonl(data, filename="pins_details.jsonl"):
    # Ensure the directory exists
    directory = "jsonl_output"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    with open(file_path, 'a', encoding='utf-8') as file:
        # Write the dictionary as a single JSON object per line
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Data saved to {file_path}")

# Save HTML to a file
def save_html_to_file(html_content, page, item):
    directory = "detailed_html"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(directory, f"page_{page}_item_{item}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)


# Function to scrape detailed pages
def scrape_details(start_page, end_page):
    driver = setup_driver()
    all_details = []

    try:
        for page in range(start_page, end_page + 1):
            print(f"Scraping Page {page}...")
            driver.get(f"https://pinandpop.com/pins?page={page}")
            time.sleep(3)  # Allow page to load

            # Find all <a> tags inside <div class="pin-card">
            pins = driver.find_elements(By.CSS_SELECTOR, "div.pin-card div.card-body h5.card-title a")
            print(f"Found {len(pins)} items on page {page}.")


            for i, pin in enumerate(pins):
                try:
                    print(f"Processing item {i + 1} on page {page}...")

                    # Get the href attribute of the link
                    href = pin.get_attribute("href")
                    if not href:
                        print(f"No href found for item {i + 1} on page {page}. Skipping.")
                        continue

                    print(f"Visiting {href}...")
                    driver.get(href)
                    time.sleep(3)  # Allow detailed page to load

                    # Get the detailed page HTML
                    detailed_html = driver.page_source

                    soup = BeautifulSoup(detailed_html, 'html.parser')
                    name = soup.find('h1', class_='h3').text.strip() if soup.find('h1', class_='h3') else None
                    print(f"Extracted Name: {name}")

                    series = soup.find('a', href=lambda x: x and '/series/' in x).text.strip() if soup.find('a', href=lambda x: x and '/series/' in x) else None
                    print(f"Extracted Series: {series}")

                    rarity_span = soup.find('span', class_=lambda c: c and 'badge' in c.split() and 'rarity' in c.split())

                    # Extract the text if the element is found
                    rarity = rarity_span.text.strip() if rarity_span else None
                    print(f"Extracted Rarity: {rarity}")

                    origin = soup.find('a', href=lambda x: x and '/origin/' in x).text.strip() if soup.find('a', href=lambda x: x and '/origin/' in x) else None
                    print(f"Extracted Origin: {origin}")

                                    # Parse the table text more robustly
                    try:
                        table = soup.find('table', class_='table table-bordered')
                        if table:
                            table_text = table.get_text(separator='\n').strip()
                            print(f"Extracted Table Text:\n{table_text}")

                            # Parse the table text into lines and clean them
                            lines = [line.strip() for line in table_text.splitlines() if line.strip()]
                            print(f"Cleaned Lines:\n{lines}")

                            # Initialize variables
                            edition = None
                            release_date = None
                            original_price = None
                            sku = None

                            # Iterate through lines to match key-value pairs
                            for i, line in enumerate(lines):
                                if line == "Edition" and i + 1 < len(lines):
                                    edition = lines[i + 1]
                                elif line == "Release Date" and i + 1 < len(lines):
                                    release_date = lines[i + 1]
                                elif line == "Original Price" and i + 1 < len(lines):
                                    original_price = lines[i + 1]
                                elif line == "SKU" and i + 1 < len(lines):
                                    sku = lines[i + 1]

                            # Print the extracted values
                            print(f"Extracted Edition: {edition}")
                            print(f"Extracted Release Date: {release_date}")
                            print(f"Extracted Original Price: {original_price}")
                            print(f"Extracted SKU: {sku}")
                        else:
                            print("Table not found in the HTML.")
                    except Exception as e:
                        print(f"Error during text-based extraction: {e}")

                    # Extract Image URL
                    img_tag = soup.find('img', class_='img-fluid m-auto')
                    img_url = img_tag['src'].strip() if img_tag and 'src' in img_tag.attrs else None
                    print(f"Extracted Image URL: {img_url}")


                    # Extract tags
                    tags = [tag.text.strip() for tag in soup.find_all('a', class_='badge rounded-pill bg-secondary')]
                    print(f"Extracted Tags: {tags}")
                    # Extract description
                    try:
                        description = None
                        for div in soup.find_all('div', class_='card-body'):
                            if div.find('h5') and 'Pin Description:' in div.find('h5').text:
                                description = div.text.replace('Pin Description:', '').strip()
                                break

                        print(f"Extracted Description: {description}")
                    except Exception as e:
                        print(f"An error occurred while extracting the description: {e}")
                    # Append extracted data to the list
                    save_to_jsonl({
                        "name": name,
                        "series": series,
                        "rarity": rarity,
                        "origin": origin,
                        "edition": edition,
                        "release_date": release_date,
                        "original_price": original_price,
                        "sku": sku,
                        "description": description,
                        "image_url": img_url,
                        "tags": tags,
                    })

                    print(f"Extracted details for item {i + 1} on page {page}.")

                    # Go back to the main page
                    driver.back()
                    time.sleep(2)

                except Exception as e:
                    print(f"Error processing item {i + 1} on page {page}: {e}")
                    driver.back()
                    time.sleep(2)




    finally:
        driver.quit()

    return all_details


# Save data to a CSV file
def save_to_file(data, filename="pins_details.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Main execution
if __name__ == "__main__":
    start_page = 10
    end_page = 3600  # Change this to your desired range
    scraped_details = scrape_details(start_page, end_page)
    save_to_file(scraped_details)
