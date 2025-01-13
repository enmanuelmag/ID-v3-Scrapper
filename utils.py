import os
import time
import random

from selenium import webdriver
from typing import Literal, List, Callable
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebElement
from selenium.webdriver.support import expected_conditions as EC

def clean_url(url: str) -> str:
    return url.split('?')[0]

def build_url_search(url_base, keywords = []):
  url = f'{url_base}?q='

  for keyword in keywords:
    url += f'{keyword}%20'
  
  return url

def parse_text_number(text: str) -> int:
    suffixes = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}

    if text is None or len(text) == 0:
        return 50

    suffix_text = text[-1].lower()
    try:
        if suffix_text in suffixes:
            number = float(text[:-1])
            multiplier = suffixes[suffix_text]
            return int(number * multiplier)
        
        return int(text)
    except ValueError:
        return 50


def clean_text(text: str) -> str:
    if text is None:
        return ''

    return text.strip().replace('\n', '').replace('\t', '')

class SearcherDriver:
    def __init__(self, file, executable_path = None, check_captcha_fn = None):
        assert file is not None, "No file provided"

        if executable_path is None:
            print('[WARNING] No executable path provided. Searching for chromedriver in PATH')

        self.file = file
        self.check_captcha_fn = check_captcha_fn
        self.by_type = {"id": By.ID, "css": By.CSS_SELECTOR, "xpath": By.XPATH}
        self.driver = webdriver.Chrome(service=webdriver.ChromeService(executable_path=executable_path))

        self.url_scraped = self.load_url_scraped()

    def append_data(self, data: str):
        with open(self.file, 'a') as f:
            f.write(data + '\n')
            self.url_scraped.append(clean_url(data.split(',')[0]))

    def is_url_scraped(self) -> bool:
        print("[DRIVER] Checking if URL is already scraped")
        result = clean_url(self.driver.current_url) in self.url_scraped

        print(f"[DRIVER] Scraped: {result} - URL: {clean_url(self.driver.current_url)}")

        return result

    def load_url_scraped(self):
        if not os.path.exists(self.file):
            print(f"[DRIVER] File not found: {self.file}")
            with open(self.file, 'w') as f:
                f.write('link	description	username	date	links	comments	shares	comments_text\n')
            return []

        with open(self.file, 'r') as f:
            lines = f.readlines()

            urls = [line.split('\t')[0] for line in lines]
            print(f"[DRIVER] Scraped URLs: {urls}")

            return urls

    def get(self, url: str):
        self.driver.get(url)

    def close(self):
        self.driver.close()

    def wait(self, timeout = 5):
        WebDriverWait(self.driver, timeout)

    def random_sleep(self, min_v = 1, max_v = 5):
        seconds = random.randint(min_v, max_v)
        print(f"[DRIVER] Sleeping for {seconds} seconds")
        time.sleep(seconds)

    def get_element_by(self, selector_type: Literal['id', 'css', 'xpath'], selector: str, timeout = 40, from_capcha=False) -> WebElement | None:
        by_selector = self.by_type.get(selector_type)

        if by_selector is None:
            raise ValueError(
                f"Invalid selector type: {selector_type} - must be 'id', 'css', or 'xpath'"
            )
        
        if self.check_captcha_fn is not None and not from_capcha:
            self.check_captcha_fn(self)

        element = None
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_selector, selector))
            )
        except Exception as e:
            pass

        if element is None and not from_capcha:
            print(f"[WARMING] Element not found: {selector}")
            return None
            # if not none_is_ok:
            #     raise Exception(f"Element not found: {selector}")

        return element


    def get_elements_by(self, selector_type: Literal['id', 'css', 'xpath'], selector: str, timeout = 40, from_capcha=False) -> List[WebElement]:
        by_selector = self.by_type.get(selector_type)

        if by_selector is None:
            raise ValueError(
                f"Invalid selector type: {selector_type} - must be 'id', 'css', or 'xpath'"
            )
        
        if self.check_captcha_fn is not None and not from_capcha:
            self.check_captcha_fn(self)

        elements = None
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((by_selector, selector))
            )
        except Exception as e:
            pass

        if elements is None and not from_capcha:
            print(f"[WARMING] Elements not found: {selector}")
            # if not none_is_ok:
            #     raise Exception(f"Elements not found: {selector}")
            # else:
            return []

        return elements
    
    def run(self, run_fn: Callable):
        try:
            print(f"[DRIVER] Running {run_fn.__name__}")
            run_fn(self)
        except Exception as e:
            raise e
