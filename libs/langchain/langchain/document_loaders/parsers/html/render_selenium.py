"""
selenium "renders" a url/filepath (string) into an html-string, loading javascript (simulating browser)

utilities for rendering html with selenium must be in their own file.
the api is specific to selenium, so method signatures must contain selenium constructs.
therefore certain imports cannot be optional due to type-hints.
"""

from typing import Literal, Optional, Union
from collections.abc import Iterator
import logging

import pathlib
from os.path import abspath

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.service import Service
from selenium.webdriver.common.options import ArgOptions

LOG = logging.getLogger(__name__)


def selenium_get_html_sequence (
    sources: Iterator[str],
    browser: Literal["chrome", "firefox"] = "chrome",
    arguments: list[str] = ["--headless", "--no-sandbox"],  ## XXX TODO 2023-11-08 @mziru how to fix this 'mutable' warning?!
) -> Iterator[tuple[str, str]]:
    """
    selenium produces a raw html text, so this returns tuple with source/systemid, otherwise it would be lost.
    """
    if isinstance(sources, str):
        sources = [sources]
    
    with selenium_get_driver(browser, arguments) as driver:
        for source in sources:
            yield source, selenium_get_html(driver, source)
    
def selenium_get_html (
    driver: WebDriver,
    source: str,
) -> str:
    driver.get(_fix_url_for_file(source))
    html = driver.page_source
    LOG.debug(f"selenium_get_html({source}):\n\t{html}")
    return html
# sadly there's no adapter from selenium.webdriver.remote.webelement.WebElement to xml.dom
# def selenium_get_dom(
#     driver: WebDriver,
#     source: str,
# ):
#     driver.get(_fix_url_for_file(source))
#
#     # jsresult: selenium.webdriver.remote.webelement.WebElement
#     jsresult = driver.execute_script(
#         # "return document")
#         "return document.documentElement")
#         # "return document.documentElement.outerHTML")
#     return jsresult


def selenium_get_driver (
    browser: Literal["chrome", "firefox"] = "chrome",
    arguments: list[str] = ["--headless", "--no-sandbox"],  ## XXX TODO 2023-11-08 @mziru how to fix this 'mutable' warning?!
    binary_location: Optional[str] = None,
    executable_path: Optional[str] = None,
) -> WebDriver: 
    '''
    Create and return a WebDriver instance based on the specified browser.
    for chrome, if 'arguments' contains "--headless", then "--no-sandbox" is also recommended.
    Args:
        
    Raises:
        ValueError: If an invalid browser is specified.
    Returns:
        WebDriver: A Chrome|Firefox instance for the specified browser.
    '''
    
    driver_class: type[WebDriver]
    service_class: type[Service]
    options_class: type[ArgOptions]
    driver_class, service_class, options_class = _selenium_get_driver_classes(browser)
    
    if "chrome"==browser and ["--headless"]==arguments:
        arguments += ["--no-sandbox"]
    
    options: ArgOptions = options_class()
    if binary_location:
        options.binary_location = binary_location
    for arg in arguments:
        options.add_argument(arg)
    
    if executable_path:
        return driver_class(
            options=options,
            service=service_class(
                executable_path=executable_path))
    else:
        return driver_class(
            options=options)
def _selenium_get_driver_classes (
    browser: Literal["chrome", "firefox"],
) -> tuple[type[WebDriver], type[Service], type[ArgOptions]]:
    '''
    Create and return a WebDriver instance based on the specified browser.
    for chrome, if 'arguments' contains "--headless", then "--no-sandbox" is also recommended.
    Args:
        
    Raises:
        ValueError: If an invalid browser is specified.
    Returns:
        WebDriver: A Chrome|Firefox instance for the specified browser.
    '''
    
    driver_class: type[WebDriver]
    service_class: type[Service]
    options_class: type[ArgOptions]
    
    match browser.lower():
        case "chrome":
            from selenium.webdriver import Chrome
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            driver_class, service_class, options_class =  Chrome, ChromeService, ChromeOptions
        case "firefox":
            from selenium.webdriver import Firefox
            from selenium.webdriver.firefox.service import Service as FirefoxService
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            driver_class, service_class, options_class = Firefox, FirefosService, FirefoxOptions
        case _:
            raise ValueError(f"Invalid browser ({browser}) specified. Use 'chrome' or 'firefox'.")
    
    return driver_class, service_class, options_class


def _fix_url_for_file (
    source: str
) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return source
    else:
        return pathlib.Path(abspath(source)).as_uri()

    
if __name__ == "__main__":
    test_sequence = [
        "test1basic.html",
        "test7javascript.html"]
    for y in selenium_get_html_sequence(test_sequence):
        print(f"{y[0]}\n{y[1]}")
    
    test_driver = "test2flat.html"
    with selenium_get_driver() as x:
        y = selenium_get_html(x, test_driver)
        print(f"{test_driver}\n{y}")
