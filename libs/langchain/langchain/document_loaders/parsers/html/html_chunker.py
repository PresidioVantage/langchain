"""
# HTML Chunking

A lightweight (SAX) HTML parse, "chunked" into sequence of contiguous text segments, each with all "relevant" headers.
Chunks are always delimited by relevant headers, but also by a (configurable) set of tags between-which to chunk.
for reference, see https://www.w3.org/WAI/tutorials/page-structure/headings/

requires lxml unless "strict" argument is set to True
uses/requires selenium if requested

## USAGE:

chunker = HtmlChunker()

def example_list():
    for x in chunker.parseList("test1basic.html"):
        print(x)
def example_queue():
    q =  chunker.parse_queue(["test2flat.html", "test3semantic.html"])
    while q
        print(q.popleft())
def example_events():
    chunker.parse_events(
        ["test4deep.html", "test5fail.html"],
        print)
def example_illformed():
    for x in chunker.parseList("test6illformed.html", true):
        print(x)
def example_illformed_urls():
    chunker.parse_events(
        ["test1basic.html", "https://en.wikipedia.org/wiki/Kurt_G%C3%B6del"],
        print)

## DEVELOPER DOCS:
This code builds 3 data structures via the classes ElemPos, Header, and ChunkPos.
An instance of each is maintained throughout (at each event of) the SAX parse.
Each class represents a tree structure, and stores only a (singly-linked list) pointer "upward" recursively to a root.
ElemPos models a tree consistent with the DOM itself.
Header permits "prior" references to be prior-siblings *in-addition-to* ancestors.
ChunkPos models a tree similar to Header, but also containing non-header "chunking" elements in the hierarchy.

(origin: https://github.com/PresidioVantage/html-chunking)
"""

from typing import (
    Callable,
    Generator,
    Literal,
    Optional,
)
from collections import deque
from collections.abc import (
    Iterable,
    Collection,
)

import logging

import xml.sax.xmlreader
import xml.sax.handler

import urllib.request
import pathlib
from os.path import abspath
from io import StringIO

import parse_html


LOG = logging.getLogger(__name__)

# CONSTANTS
ALL_HEADER_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6"]

# CONFIGS
DEFAULT_CHUNK_TAGS = [
    "div", "p", "blockquote", "ol", "ul"]  # TODO consider adding "article" to this list?
TUPLES_NOT_DICT = False
FLATTEN_TAGS = [
    "header", "hgroup"]
DEFAULT_STRICT = True

class HtmlChunker:
    """
    for single-threaded, sequential parsing only.

    this uses lxml.html parser to tolerate html's lax tag structure,
    which significantly degrades performance (including a full DOM parse).

    the "strict" argument controls whether to use built-in xml.sax instead of lxml.html+lxml.sax
    this is lighter and faster but requires the document have balanced tags (~well-formed xml)

    the "source" argument is passed directly to the parser (with exceptions):
    lxml.html for loose parsing:
        filename_or_url: "a filename, URL, or file-like object"
        (https://lxml.de/apidoc/lxml.html.html#lxml.html.parse)
        http/https URLs are not supported by lxml, so they are converted to http connections (IO)
    xml.sax for strict parsing:
        filename_or_stream: "can be a filename or a file object"
        (https://docs.python.org/3/library/xml.sax.html)
    """
    
    
    
    
    def __init__(
        self,
        parse_render: Literal["xml", "lxml", "selenium"] = "lxml",
        header_tags: Optional[Collection[str]] = None,
        chunk_tags: Optional[Collection[str]] = None,
    ):
        
        self.parse_render: Literal["xml", "lxml", "selenium"] = parse_render.lower()
        
        if header_tags is None:
            header_tags = ALL_HEADER_TAGS.copy()
        elif not all(x in ALL_HEADER_TAGS for x in header_tags):
            raise Exception(f"invalid header_tags: {header_tags}")
        self.header_tags: Collection[str] = header_tags
        
        if chunk_tags is None:
            chunk_tags = DEFAULT_CHUNK_TAGS.copy()
        if any(x in ALL_HEADER_TAGS for x in chunk_tags):
            raise Exception(f"invalid chunk_tags: {chunk_tags}")
        self.chunk_tags: Collection[str] = chunk_tags

    def parse_chunk_sequence(
        self,
        sources: Iterable[any],
    ) -> Generator[dict[str, any], None, None]:
        for q in self._parse_queue_sequence(sources):
            yield from q
    def _parse_queue_sequence(
        self,
        sources: Iterable[any],
    ) -> Generator[deque[dict[str, any]], None, None]:
        for source in sources:
            yield self.parse_queue(source)

    def parse_queue(
        self,
        source: any,
    ) -> deque[dict[str, any], None, None]:

        the_q = deque()
        self.parse_events([source], the_q.append)
        return the_q
    
    def parse_events(
        self,
        sources: Iterable[any],
        callback: Callable[[dict[str, any]], any],
    ):

        handler = self._new_handler(callback)
        
        match self.parse_render:
            case "xml":
                for source in sources:
                    parse_html.xml_get_sax(source, handler)
            case "lxml":
                for source in sources:
                    parse_html.lxml_get_sax(source, handler)
            case "selenium":
                parse_html.selenium_get_sax(sources, handler)
            case _:
                raise ValueError(f"Invalid parse_render ({self.parse_render}) specified. Use 'xml', 'lxml' or 'selenium'.")

    # Helper Methods:

    def _new_handler(
        self,
        callback: Callable[[dict[str, any]], any]
    ) -> xml.sax.handler.ContentHandler:

        return ChunkHandler(
            callback,
            self.header_tags,
            self.chunk_tags)


class ElemPos:
    """
    an HTML Element node, including:
      a pointer to its Parent Element (or None if root element)
      a pointer to its "Nearest" Header (or None)

    'parent' is a singly-linked-list to the root.
    'head' is a chain of self-or-prior-sibling headers, "nearest by level."
    'tag' is the the only other "semantic" data.
    'index' and 'child_elem_count' are merely for identification/debugging purposes.
    """

    def __init__(
        self,
        parent: "ElemPos",
        tag: str,
        head: "Header",
    ):

        if not tag:
            raise Exception(f"ElemPos({tag})")

        self.parent: ElemPos = parent
        self.tag: str = tag
        self.head: Header = head
        self.index: int = 1
        # mutable state, gets incremented by every child __init__:
        self.child_elem_count: int = 0

        if self.parent:
            self.parent.child_elem_count = 1 + self.parent.child_elem_count
            self.index = self.parent.child_elem_count

    def get_meta(self):
        return self.head.get_dict() if self.head else []

    # these functions only used for logging/debugging:
    def __str__(self):
        return f"{self.get_indent()}E{self.get_identifier()}"

    def get_depth(self):
        return 1 + (self.parent.get_depth() if self.parent else 0)

    def get_indent(self, offset: int = 0):
        return "  " * (offset + self.get_depth())

    def get_identifier(self):
        return (self.parent.get_identifier() if self.parent else "") \
            + f"[{self.index}]{self.tag}"

    def get_xpath(self):
        return (self.parent.get_xpath() if self.parent else "") \
            + f"/*[{self.index}]"


class Header:
    """
    an HTML Header Element node (i.e. H1 ... H6), including:
      a pointer to its "Nearest Prior Higher" Header (or None if there is none)
    """

    def __init__(
        self,
        prior_higher: "Header",
        elem: ElemPos
    ):

        if elem is None:
            raise Exception("Header(None)")
        if elem.tag not in ALL_HEADER_TAGS:
            raise Exception(f"bad header tag {elem.tag}")

        self.prior_higher: Header = prior_higher
        self.pos: ElemPos = elem
        self.level: int = int(elem.tag[1])
        self.depth: int = 1
        # mutable state, appended-to until header-tag is closed:
        self.text: str = ""

        # supersede prior if is a sibling and lower-level (higher number)
        while \
                self.prior_higher and \
                self.prior_higher.pos.parent == self.pos.parent and \
                self.prior_higher.level >= self.level:
            self.prior_higher = self.prior_higher.prior_higher

        if self.prior_higher:
            self.depth = self.prior_higher.depth + (self.prior_higher.level >= self.level)
        else:
            self.depth = 1

    def get_dict(self):
        return (self.prior_higher.get_dict() if self.prior_higher else []) \
            + [self.get_val()]

    def get_val(self):
        return (f"D{self.depth} " if 1 < self.depth and not TUPLES_NOT_DICT else "") + self.pos.tag, self.text

    # these functions only used for logging/debugging:
    def __str__(self):
        return f"{self.pos.get_indent(1)}H{(self.get_header_path(), self.text)}"
        # return f"H{self.get_dict()}";

    def get_header_path(self):
        return (self.prior_higher.get_header_path() if self.prior_higher else "") \
            + ("~" if self.prior_higher and self.prior_higher.depth == self.depth else " / ") \
            + self.pos.tag


class ChunkPos:
    """
    a pointer to trace all "applicable" chunking elements (for a given position).
    this is informational-only, and it is not used-by / needed-for the actual chunking logic.
    """

    def __init__(
        self,
        parent_chunk: "ChunkPos",
        elem: ElemPos
    ):

        if elem is None:
            raise Exception("ChunkPos(elem=None)")

        self.par: ChunkPos = parent_chunk
        self.pos: ElemPos = elem

        # if this is a header, and chunk-parent is a prior-sibling, we're both Headers and
        # chunk-parent must have header-level less than this header-level.
        if self.pos.tag in ALL_HEADER_TAGS:
            while \
                    self.par and \
                    self.par.pos.parent == self.pos.parent and \
                    self.par.pos.head.level >= self.pos.head.level:
                self.par = self.par.par

    def get_chunk_path(self):
        return (self.par.get_chunk_path() + "/" if self.par else "") \
            + self.pos.tag

    def __str__(self):
        return f"{self.pos.get_indent(1)}C({self.get_chunk_path()})"


class ChunkHandler(xml.sax.handler.ContentHandler):
    """
    for single-threaded, sequential parsing only.
    """

    def __init__(
        self,
        yield_function: Callable[[dict], any] = lambda x: None,
        header_tags: Optional[Collection[str]] = None,
        chunk_tags: Optional[Collection[str]] = None
    ):

        super().__init__()
        
        if chunk_tags is None:
            chunk_tags = DEFAULT_CHUNK_TAGS.copy()
        elif not all(x in ALL_HEADER_TAGS for x in header_tags):
            raise Exception(f"invalid header_tags: {header_tags}")
        
        if header_tags is None:
            header_tags = ALL_HEADER_TAGS.copy()
        elif any(x in ALL_HEADER_TAGS for x in chunk_tags):
            raise Exception(f"invalid chunk_tags: {chunk_tags}")

        self.yield_function: Callable[[dict], any] = yield_function
        self.header_tags: Collection[str] = header_tags
        self.chunk_tags: Collection[str] = chunk_tags

        # mutable state, esp. tracking pointers between/during parse events:
        self.uri: Optional[str] = None
        self.current: Optional[ElemPos] = None
        self.prior_headers: Optional[Header] = None
        self.header_sent: bool = False
        self.chunk: Optional[ChunkPos] = None
        self.text: Optional[str] = None
        self.building_header: Optional[Header] = None

    # ignore namespace information
    def startElementNS(self, nsname: tuple[str, str], qname, attrs):
        self.startElement(nsname[1], attrs)

    def endElementNS(self, nsname: tuple[str, str], qname):
        self.endElement(nsname[1])

    def setDocumentLocator(self, loc: xml.sax.xmlreader.Locator):
        # xml.sax does this when given an open URLConnection, unfortunately
        if loc is None or loc.getSystemId() is None:
            LOG.warning(f"suppressed setDocumentLocator(None), maintained {self.uri} ({loc})")
        else:
            LOG.debug(f"setDocumentLocator({loc.getSystemId()}) replacing {self.uri}")

            self.uri = loc.getSystemId()

    def startDocument(self):
        LOG.debug("startDocument()")

    def endDocument(self):
        LOG.debug("endDocument()")
        self.uri = None

    def startElement(self, name: str, attrs):
        if name.lower() in FLATTEN_TAGS:
            LOG.debug(f"<FLATTEN '{name}'")
            return

        self.current = ElemPos(self.current, name.lower(), self.prior_headers)
        LOG.debug(f"<{self.current}")

        # if already actively building a header, no other logic (except one validation)
        if self.building_header:
            if self.current.tag in self.header_tags:
                raise Exception(f"start {name} within {self.building_header}")

        else:
            if self.current.tag in self.header_tags:
                self.building_header = Header(self.prior_headers, self.current)

                # if this header supersedes the prior,
                # and the prior-header hasn't been sent yet,
                # then force it to be sent even if empty.
                force_chunk = \
                    not self.header_sent and \
                    self.prior_headers != self.building_header.prior_higher
                self.send_chunk(force_chunk)

                self.prior_headers = self.building_header
                self.current.head = self.building_header
                self.header_sent = False

                if self.chunk:
                    self.chunk = ChunkPos(self.chunk, self.current)
                    LOG.debug(f"#{self.chunk}")

            # a header tag which is not-captured is still a helpful place to slice the current ChunkPos
            elif self.current.tag in ALL_HEADER_TAGS:
                self.send_chunk()

            elif self.current.tag in self.chunk_tags:
                if not self.send_chunk():
                    self.text = ""

                self.chunk = ChunkPos(self.chunk, self.current)
                LOG.debug(f"#{self.chunk}")

    def endElement(self, name: str):
        if name.lower() in FLATTEN_TAGS:
            LOG.debug(f"/FLATTEN '{name}'")
            return

        if self.current is None:
            raise Exception(f"end <{name}> in None")
        if name != self.current.tag:
            raise Exception(f"end <{name}> in <{self.current.tag}>")
        LOG.debug(f"/{self.current}")

        if self.building_header:
            # building_header mode does nothing, except flag when it is over
            if self.current == self.building_header.pos:
                self.building_header.text = self.building_header.text.strip()
                LOG.debug(f"#{self.building_header}")
                self.building_header = None

        else:
            # this tag ends an element which is not a Header itself

            if self.prior_headers != self.current.head:
                self.send_chunk(not self.header_sent)

                self.prior_headers = self.current.head
                while self.chunk and self.chunk.pos.parent == self.current:
                    self.chunk = self.chunk.par
                if self.chunk is None:
                    self.text = None

            if self.current.tag in self.chunk_tags:
                self.send_chunk()

                end_chunk = self.chunk
                while end_chunk.pos != self.current:
                    end_chunk = end_chunk.par
                self.chunk = end_chunk.par
                if self.chunk is None:
                    self.text = None

        self.current = self.current.parent

    def characters(self, content: str):
        # if content.strip():
        if self.current is None:
            raise Exception(f"characters in None: {content}")

        LOG.debug("T" + self.current.get_indent(1) + content.strip())

        if self.building_header:
            self.building_header.text += content

        elif self.chunk:
            self.text += content

    def send_chunk(
        self,
        send_blank = False
    ) -> Optional[dict[str, any]]:
        LOG.debug(f"send_chunk({'!' if send_blank else ''}{self.chunk})")
        if self.chunk:
            retval = {
                "uri": self.uri,
                "pos": self.chunk.pos.get_identifier(),
                "text": self.text.strip(),
                "meta": self.chunk.pos.get_meta() if TUPLES_NOT_DICT else dict(self.chunk.pos.get_meta())}

            if send_blank or retval["text"]:
                self.yield_function(retval)
                self.header_sent = True

            self.text = ""

            return retval
        else:
            return None


if __name__ == "__main__":
    import sys

    main_verbose = False
    
    # main_parse_render = "xml"
    main_parse_render = "lxml"
    # main_parse_render = "selenium"
    
    # if no sources are specified by command-line args, use defaults
    main_sources = sys.argv[1:] if 1 < len(sys.argv) else [
        "test1basic.html",
        # "https://presidiovantage.com/html_chunker/test1basic.html",
        # "https://en.wikipedia.org/wiki/Kurt_G%C3%B6del",
        # "https://plato.stanford.edu/entries/goedel/",
    ]
    
    if main_verbose:
        logging.basicConfig(level=logging.DEBUG)
        main_callback = LOG.info
    else:
        main_callback = print

    chunker = HtmlChunker(main_parse_render)
    chunker.parse_events(main_sources, main_callback)
