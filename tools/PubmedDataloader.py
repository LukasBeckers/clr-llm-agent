import requests
import time
import os


class PubmedDataloader():
    def __init__(self, api_url):
        """
        Initializes the APIClient with the base URL of the backend API.

        :param api_url: Base URL of the backend API (e.g., "http://localhost:5000")
        """
        self.api_url = api_url.rstrip('/')

    def start_loading(self, search_term, email, max_results=None):
        """
        Initiates the loading of articles based on the search term.

        :param search_term: The term to search for articles.
        :param email: User's email address.
        :param max_results: Maximum number of results to fetch (optional).
        :return: loader_id to track the loading process.
        :raises: requests.HTTPError if the request fails.
        """
        url = f"{self.api_url}/api/start"
        payload = {
            "search_term": search_term,
            "email": email
        }
        if max_results is not None:
            payload["max_results"] = max_results

        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        data = response.json()
        loader_id = data.get("loader_id")
        if not loader_id:
            raise ValueError("No loader_id returned from API")
        return loader_id

    def get_status(self, loader_id):
        """
        Retrieves the current status of the loading process.

        :param loader_id: The unique identifier for the loader.
        :return: A dictionary containing the progress information.
        :raises: requests.HTTPError if the request fails.
        """
        url = f"{self.api_url}/api/status"
        params = {"loader_id": loader_id}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def download_file(self, loader_id, file_type, download_dir='downloads'):
        """
        Downloads the specified file type (JSON or ZIP) associated with the loader_id.

        :param loader_id: The unique identifier for the loader.
        :param file_type: Type of the file to download ('json' or 'zip').
        :param download_dir: Directory where the downloaded file will be saved.
        :return: Path to the downloaded file.
        :raises: ValueError for invalid file_type.
                 requests.HTTPError if the download request fails.
        """
        if file_type not in ['json', 'zip']:
            raise ValueError("file_type must be 'json' or 'zip'")

        url = f"{self.api_url}/api/download/{file_type}"
        params = {"loader_id": loader_id}
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()

        os.makedirs(download_dir, exist_ok=True)
        filename = f"{loader_id}.{file_type}"
        filepath = os.path.join(download_dir, filename)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return filepath

    def download_json_as_dict(self, loader_id):
        """
        Downloads the JSON data associated with the loader_id and loads it into a Python dictionary.

        :param loader_id: The unique identifier for the loader.
        :return: Python dictionary containing the JSON data.
        :raises: ValueError if the loader_id is invalid or JSON parsing fails.
                 requests.HTTPError if the download request fails.
        """
        url = f"{self.api_url}/api/download/json"
        params = {"loader_id": loader_id}
        response = requests.get(url, params=params)
        response.raise_for_status()

        try:
            json_data = response.json()
        except ValueError:
            raise ValueError("Failed to parse JSON response")

        return json_data

    def search_and_download(self, search_string, email, max_results=None,
                            poll_interval=5, timeout=3600):
        """
        Performs the entire workflow: starts the search, monitors progress, and downloads the results.

        :param search_term: The term to search for articles.
        :param email: User's email address.
        :param max_results: Maximum number of results to fetch (optional).
        :param download_dir: Directory where the downloaded files will be saved.
        :param poll_interval: Time in seconds between status checks.
        :param timeout: Maximum time in seconds to wait for the process to complete.
        :param fetch_json_in_memory: If True, load JSON data into memory as a dict instead of saving to disk.
        :return: 
            If fetch_json_in_memory is True:
                Dictionary with JSON data and path to the downloaded ZIP file.
            Else:
                Dictionary with paths to the downloaded JSON and ZIP files.
        :raises: TimeoutError if the process takes longer than the specified timeout.
                 Exception if the loading process encounters an error.
        """
        loader_id = self.start_loading(search_string, email, max_results)
        print(f"Started loading with loader_id: {loader_id}")

        start_time = time.time()
        while True:
            status = self.get_status(loader_id)
            print(f"Status: {status}")

            # Adjust the condition based on your actual API response structure
            if status.get("status") == "Completed":
                print("Loading completed.")
                break
            elif status.get("status") == "error":
                raise Exception("Loading failed with error.")
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    raise TimeoutError("Loading timed out.")
                print(f"Loading in progress... waiting for {poll_interval} seconds.")
                time.sleep(poll_interval)

        result = {}
       
        json_data = self.download_json_as_dict(loader_id)
        result['json'] = json_data
        print("JSON data loaded into memory.")

        return result
