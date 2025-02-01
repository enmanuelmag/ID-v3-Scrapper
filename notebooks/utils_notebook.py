import os
import time
import random
import re
import pandas as pd

from selenium import webdriver
from typing import Literal, List, Callable
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebElement
from selenium.webdriver.support import expected_conditions as EC


def str_to_date_tiktok(text: str, pivot_date="2025-01-30") -> str:
    text = str(text).strip()

    def _parse_date(_text: str) -> str:
        formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"]
        for fmt in formats:
            try:
                str_date = pd.to_datetime(_text, format=fmt, errors="raise")
                return str_date.strftime("%Y-%m-%d")
            except ValueError:
                continue

    if text and re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return _parse_date(text)
    elif text and re.match(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", text):
        return _parse_date(text)
    elif text and re.match(r"\d{1,2}[-/]\d{1,2}", text):
        if "-" in text:
            _, month = text.split("-")
        else:
            _, month = text.split("/")
        if month == 1:
            text = f"{text}-2025"
        else:
            text = f"{text}-2024"
        text = text.replace("/", "-")
        str_date = pd.to_datetime(text, format="%m-%d-%Y", errors="raise")
        return str_date.strftime("%Y-%m-%d")
    elif "ago" in text:
        pivot_date = pd.to_datetime(pivot_date, errors="raise", format="%Y-%m-%d %H:%M:%S")
        if "d" in text:
            date = pivot_date - pd.Timedelta(days=int(text.split("d ago")[0]))
        elif "h" in text:
            date = pivot_date - pd.Timedelta(hours=int(text.split("h ago")[0]))
        elif "w" in text:
            date = pivot_date - pd.Timedelta(weeks=int(text.split("w ago")[0]))
        elif "m" in text:
            date = pivot_date - pd.Timedelta(month=int(text.split("m ago")[0]))
        else:
            date = pivot_date
        return date.strftime("%Y-%m-%d")

    return None


def clean_url(url: str) -> str:
    return url.split("?")[0]


def build_url_search(url_base, keywords=[]):
    url = f"{url_base}?q="

    for keyword in keywords:
        url += f"{keyword}%20"

    return url


def parse_text_number(text: str) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

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
        return ""

    return text.strip().replace("\n", "").replace("\t", "")


class SearcherDriver:
    def __init__(self, executable_path=None, check_captcha_fn=None):
        if executable_path is None:
            print(
                "[WARNING] No executable path provided. Searching for chromedriver in PATH"
            )

        # self.file = file
        self.check_captcha_fn = check_captcha_fn
        self.by_type = {"id": By.ID, "css": By.CSS_SELECTOR, "xpath": By.XPATH}
        self.driver = webdriver.Chrome(
            service=webdriver.ChromeService(executable_path=executable_path)
        )

        # self.url_scraped = self.load_url_scraped()

    def set_file(self, file: str, city: str):
        self.file = file
        self.city = city
        self.url_scraped = self.load_url_scraped()

    def append_data(self, data: str):
        with open(self.file, "a") as f:
            f.write(f"{data}\t{self.city}\n")
            self.url_scraped.append(clean_url(data.split(",")[0]))

    def is_url_scraped(self) -> bool:
        result = clean_url(self.driver.current_url) in self.url_scraped

        print(f"[DRIVER] Scraped: {result}")

        return result

    def load_url_scraped(self: str):
        if not os.path.exists(self.file):
            print(f"[DRIVER] File not found: {self.file} - Creating new file")
            with open(self.file, "w") as f:
                f.write(
                    "link	description	username	date	links	comments	shares	comments_text	scrapped_at	city\n"
                )
            return []

        with open(self.file, "r") as f:
            lines = f.readlines()

            urls = [line.split("\t")[0] for line in lines]
            print(f"[DRIVER] Scraped URLs: {len(urls)}")

            return urls

    def get(self, url: str):
        self.driver.get(url)

    def close(self):
        self.driver.close()

    def wait(self, timeout=5):
        WebDriverWait(self.driver, timeout)

    def random_sleep(self, min_v=1, max_v=5):
        seconds = random.randint(min_v, max_v)
        #print(f"[DRIVER] Sleeping for {seconds} seconds")
        time.sleep(seconds)

    def get_element_by(
        self,
        selector_type: Literal["id", "css", "xpath"],
        selector: str,
        timeout=40,
        from_capcha=False,
    ) -> WebElement | None:
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

    def get_elements_by(
        self,
        selector_type: Literal["id", "css", "xpath"],
        selector: str,
        timeout=40,
        from_capcha=False,
    ) -> List[WebElement]:
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
            #print(f"[DRIVER] Running {run_fn.__name__}")
            run_fn(self)
        except Exception as e:
            raise e
