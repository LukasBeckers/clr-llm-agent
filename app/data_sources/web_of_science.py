from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import pandas as pd

from data_sources.data_source_interface import IDataSource


class WebOfScienceScraper(IDataSource):

    basic_search_url = "https://www.webofscience.com/wos/woscc/basic-search"
    advanced_serach_url = "https://www.webofscience.com/wos/woscc/advanced-search"

    def __init__(self):
        self.ip = None # Maybe I will implement something to rotate the IP address
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    def _start_basic_serach(self):
        #search_box = driver.get()
        print(self.basic_search_url)

    def search(self, query: Union[str, List[str]]) -> DataFrame:
        return pd.DataFrame()
