"""
contains utilities for parsing html into common structures e.g. dom/sax/etree.
methods are library-specific, and the libraries are imported optionally/lazily.

most libraries are light-weight (to initialize), and methods are single-source.
selenium is heavy-weight, so the methods take a sequence of sources.
"""
import urllib
from typing import Optional, Callable, Literal
from collections.abc import Iterator
from io import StringIO
import logging

import xml.sax
from xml.sax.handler import ContentHandler
from xml.sax.xmlreader import Locator

LOG = logging.getLogger(__name__)


def xml_get_sax(
    source: any,
    handler: ContentHandler,
):
    # xml.sax sets null url for open HttpConnections
    # (it works fine for String paths/urls and FileIO)
    if not isinstance(source, str) and getattr(source, "url", None):
        handler.setDocumentLocator(SystemIdLocator(source.url))
    
    xml.sax.parse(source, handler)


# **LXML**

def lxml_get_sax (
    source: any,
    handler: ContentHandler,
):
    """
    requires lxml
    """
    import lxml.sax
    
    # lxml sax has zero support for DocumentLocator
    system_id: Optional[str] = None
    if isinstance(source, str):
        system_id = source
    elif getattr(source, "url", None):
        system_id = source.url
    elif getattr(source, "name", None):
        system_id = source.name
    
    system_id: Optional[str] = get_system_id(source)
    if system_id:
        handler.setDocumentLocator(SystemIdLocator(system_id))
    
    tree = _lxml_get_etree(source)  # TODO type-hint
    lxml.sax.saxify(tree, handler)

## TODO publicize once return type is resolved
def _lxml_get_etree (
    source: any,
) -> any:  # TODO best etree(lxml.html) return type?!
    """
    requires lxml
    """
    import lxml.html
    
    # lxml does not support http references?! (despite documentation otherwise)
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        with urllib.request.urlopen(source) as c:
            tree = lxml.html.parse(c)
    else:
        tree = lxml.html.parse(source)
    
    LOG.debug(f"lxml_get_etree({source}): {tree}\n\t{tree.docinfo.URL}")
    
    return tree


# **Selenium**

def selenium_get_sax(
    sources: Iterator[str],
    handler: ContentHandler,
    browser: Literal["chrome", "firefox"] = "chrome",
    args: tuple[str] = ("--headless", "--no-sandbox"),
    # saxparser: Callable[[any, ContentHandler], None] = lxml_get_sax,
    # saxparser: Literal[lxml_get_sax, xml_get_sax] = lxml_get_sax,
):
    """
    requires selenium and lxml
    """
    from langchain.document_loaders.parsers.html.render_selenium import selenium_get_html_sequence
    
    for source, html in selenium_get_html_sequence(sources, browser, args):
        handler.setDocumentLocator(SystemIdLocator(source))
        rendered_source = StringIO(html)
        lxml_get_sax(rendered_source, handler)
        # saxparser(rendered_source, handler)


# **Utilities**

def get_system_id (
    source: any
) -> Optional[str]:
    
    system_id: Optional[str] = None
    
    # string represents a url or filename
    if isinstance(source, str):
        system_id = source
    # e.g. a url-connection e.g. urllib.request.urlopen(<url>)
    elif getattr(source, "url", None):
        system_id = source.url
    # e.g. a file e.g. open(<filename>, "r")
    elif getattr(source, "name", None):
        system_id = source.name
    
    return system_id

class SystemIdLocator(Locator):
    """
    simple adapter to obtain a Locator from a system-id alone
    """
    def __init__(self, system_id):
        self.system_id = system_id
    def getSystemId(self):
        return self.system_id

class CallbackHandler(ContentHandler):
    """
    primarily for testing/debugging
    """
    
    def __init__(
        self,
        callback: Callable[tuple, any],
        whitespace: bool = False
    ):
        self.callback: Callable[tuple, any] = callback
        self.whitespace: bool = whitespace
    
    def setDocumentLocator(self, loc: Locator):
        self.callback(
            ('setDocumentLocator', loc))
    def startDocument(self):
        self.callback(
            ('startDocument',))
    def endDocument(self):
        self.callback(
            ('endDocument',))
    
    def startElement(self, name: str, attrs):
        self.callback(
            ('startElement', name, attrs._attrs))
    def endElement(self, name: str):
        self.callback(
            ('endElement', name))
    
    def startElementNS(self, nsname: tuple[str, str], qname: str, attrs):
        self.callback(
            ('startElementNS', nsname, qname, attrs._attrs))
    def endElementNS(self, nsname: tuple[str, str], qname: str):
        self.callback(
            ('endElementNS', nsname, qname))
    
    def characters(self, text: str):
        if self.whitespace or text.strip():
            self.callback(
                ('characters', text))


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # lxml_get_sax("test7javascript.html", CallbackHandler(print))
    selenium_get_sax("test7javascript.html", CallbackHandler(print), "chrome")

