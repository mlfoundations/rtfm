import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Union

import chromedriver_binary
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from ucimlrepo.fetch import *

from rtfm.utils import initialize_dir


@dataclass
class UCIDatasetContainer:
    """Container for the results of fetching a UCI dataset"""

    variables: pd.DataFrame
    data_file: str
    metadata: Dict[str, Any]


def fetch_available_datasets(
    filter: Optional[str] = None, search: Optional[str] = None
):
    """
    Hacked version of ucimlrepo.fetch.list_available_datasets() that returns the dataset objects.
    """

    # validate filter input
    if filter:
        if type(filter) != str:
            raise ValueError("Filter must be a string")
        elif filter.lower() not in VALID_FILTERS:
            raise ValueError(
                "Filter not recognized. Valid filters: [{}]".format(
                    ", ".join(VALID_FILTERS)
                )
            )
        filter = filter.lower()

    # validate search input
    if search:
        if type(search) != str:
            raise ValueError("Search query must be a string")
        search = search.lower()

    # construct endpoint URL
    api_list_url = API_LIST_URL
    query_params = {}
    if filter:
        query_params["filter"] = filter
    else:
        query_params["filter"] = "python"  # default filter should be 'python'
    if search:
        query_params["search"] = search

    api_list_url += "?" + urllib.parse.urlencode(query_params)

    # fetch list of datasets from API
    data = None
    try:
        response = urllib.request.urlopen(api_list_url)
        data = json.load(response)["data"]
    except (urllib.error.URLError, urllib.error.HTTPError):
        raise ConnectionError("Error connecting to server")

    if len(data) == 0:
        print("No datasets found")
        return

    return data


def extract_dataset_characteristics(soup, parent_text: str, dataset_id=None):
    """Extract different fields inside the 'Dataset Characteristics' page block."""
    # Find the <h1> tag with the specific text
    h1_tag = soup.find("h1", string=parent_text)

    # Check if the <h1> tag is found
    if h1_tag:
        # Find the next sibling that is a <p> tag
        p_tag = h1_tag.find_next_sibling("p")

        # Extract the text if the <p> tag is found
        if p_tag:
            text = p_tag.get_text()
            # print(f"Found {parent_text} value: {text}")
            return text

        else:
            logging.warning(
                f"No <p> tag found after the `{parent_text}` <h1> tag for dataset {dataset_id}."
            )
    else:
        logging.warning(
            f"No <h1> tag with the specified text `{parent_text}` found for dataset {dataset_id}."
        )


def download_data_file(
    soup, results_dir: str, base_url="https://archive.ics.uci.edu"
) -> Union[str, None]:
    """Download the file containing the UCI dataset data."""
    download_link = soup.find("a", class_="btn-primary btn w-full text-primary-content")

    # Check if the 'Download' link is found
    if download_link:
        # Extract the href attribute for the download URL
        download_url = download_link.get("href")

        # Check if the URL is valid
        if download_url:
            # Form the complete URL if necessary (if the URL is relative)
            #             base_url = 'http://example.com'  # Replace with the actual base URL of the website
            if download_url.startswith("/"):
                download_url = base_url + download_url

            logging.info(f"downloading from {download_url}")

            # Download the file
            response = requests.get(download_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the file
                filename = download_link.get("download")
                if not filename:
                    filename = os.path.basename(download_url)
                initialize_dir(results_dir)
                local_results_file = os.path.join(results_dir, filename)
                with open(local_results_file, "wb") as file:
                    file.write(response.content)

                logging.info(
                    f"data file downloaded successfully to {local_results_file}"
                )
                return local_results_file
            else:
                raise requests.exceptions.RequestException(
                    f"Failed to download the file from {download_url}."
                )
        else:
            logging.error("Download URL not found.")
    else:
        logging.error("No 'Download' link found.")


def extract_h1_section_content(soup, parent_text, dataset_id=None):
    """Extract different fields under the collapsible headers ('Additional Variable Information', etc.)"""
    # Find the <h1> tag with the specific text
    h1_tag = soup.find("h1", string=parent_text)

    # Check if the <h1> tag is found
    if h1_tag:
        # Navigate to the desired <p> tag
        p_tag = h1_tag.find_next("p", class_="whitespace-pre-wrap svelte-17wf9gp")

        # Extract the text if the <p> tag is found
        if p_tag:
            text = p_tag.get_text()
            return text
        else:
            logging.warning(
                f"No matching <p> tag found in section {parent_text} for dataset {dataset_id}."
            )
    else:
        logging.warning(
            f"No <h1> tag with the specified text `{parent_text}` found for dataset {dataset_id}."
        )


def initialize_webdriver():
    # Setup Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    options.add_argument(
        "--no-sandbox"
    )  # Required if running as root user. Do not use with untrusted code
    options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems

    driver = webdriver.Chrome(options=options)
    return driver


def fetch_uci_data(
    url: str = None,
    dataset_id: int = None,
    output_dir="./ucidata",
) -> UCIDatasetContainer:
    """Fetch the data associated with a UCI dataset.

    This fetches the data files, but also scrapes the metadata from the page.
    """
    assert url or dataset_id, "must provide either url or dataset_id."
    if dataset_id is not None:
        url = f"https://archive.ics.uci.edu/dataset/{dataset_id}/"
        logging.info(f"constructed the following url from the provided ID {url}")
    else:
        dataset_id = int(
            url.replace("https://archive.ics.uci.edu/dataset/", "").split("/")[0]
        )
        logging.info(
            f"extracted dataset dataset_id {dataset_id} from the provided url {url}"
        )

    dataset_output_dir = os.path.join(output_dir, str(dataset_id))

    driver = initialize_webdriver()
    logging.info(f"fetching content from {url}")
    driver.get(url)

    # Initialize DataFrame to store data
    variables_df = pd.DataFrame()
    total_num_variables_extracted = 0

    metadata = {}

    # Loop to handle pagination. UCI displays the variables table only in
    # rows of 11 rows at a time; we load these rows and then click the
    # corresponding button (after a wait)
    while True:
        # Attempt to accept cookies if the notification is present
        try:
            WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(.,'Accept')]"))
            ).click()
            logging.debug("Accepted cookies")
        except Exception as e:
            logging.debug(
                f"No cookie notification or already accepted for dataset {dataset_id}"
            )

        # Parse table with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Download the data
        local_data_file = download_data_file(soup, dataset_output_dir)

        # Extract dataset characteristics
        for field in (
            "Dataset Characteristics",
            "Subject Area",
            "Associated Tasks",
            "Feature Type",
            "# Instances",
            "# Features",
        ):
            if field not in metadata:
                metadata[field] = extract_dataset_characteristics(
                    soup, field, dataset_id
                )

        # Extract variable info
        for field in (
            "Dataset Information",
            "Additional Variable Information",
            "Introductory Paper",
            "Papers Citing this Dataset",
        ):
            if field not in metadata:
                metadata[field] = extract_h1_section_content(soup, field, dataset_id)
                # print(f"variable info is: {metadata[field]}")

        # Parse the variables table

        table = soup.find("table", {"class": "w-full"})
        if table:
            rows = table.find_all("tr")
            total_num_variables_extracted += len(rows)
            logging.info(
                f"got {len(rows)} rows for dataset {dataset_id}; "
                f"current total: {total_num_variables_extracted}"
            )

            # Extract data from each row and append to DataFrame
            for row in rows[1:]:
                cols = row.find_all("td")
                cols = [ele.text.strip() for ele in cols]
                variables_df = variables_df.append([cols], ignore_index=True)

            # After accepting cookies, wait for a few seconds
            time.sleep(2 + np.abs(np.random.normal()))

        # Scroll to the 'Next Page' button
        try:
            next_button = driver.find_element(
                By.XPATH, "//button[@aria-label='Next Page']"
            )
            driver.execute_script("arguments[0].scrollIntoView();", next_button)

            logging.debug(f"clicking next button for dataset {dataset_id}")
            next_button.click()
            time.sleep(2 + np.abs(np.random.normal()))  # Wait for page to load
        except Exception as e:
            #             print(f"exception: {e}")
            logging.debug(
                f"No more pages to navigate or exception raised for dataset {dataset_id}"
            )
            break

    # Close Selenium WebDriver
    driver.quit()

    # Write the variables table to a CSV.
    if len(variables_df):
        variables_df.columns = [
            "Variable Name",
            "Role",
            "Type",
            "Demographic",
            "Description",
            "Units",
            "Missing Values",
        ]
        variables_df_path = os.path.join(
            dataset_output_dir, f"variables_{dataset_id}.csv"
        )
        variables_df.to_csv(variables_df_path, index=False)
        print(f"[INFO] variables table written to {variables_df_path}")
    else:
        print(
            f"[INFO] no variables table found for dataset {dataset_id} at url {url}"
            f" (this may not be due to an error or bug; some datasets have no variables info!)"
        )

    # Write the metadata to a JSON file.
    if len(metadata):
        meta_json_path = os.path.join(dataset_output_dir, f"metadata_{dataset_id}.json")
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"[INFO] metadata written to {meta_json_path}")
    else:
        print(
            f"[INFO] no metadata for dataset {dataset_id} at url {url};"
            "this is unexpected and could be the result of a bug."
        )

    return UCIDatasetContainer(
        variables=variables_df, data_file=local_data_file, metadata=metadata
    )
