from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredHTMLLoader(UnstructuredFileLoader):
    """Load `HTML` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain.document_loaders import UnstructuredHTMLLoader

    loader = UnstructuredHTMLLoader(
        "example.html", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-html
    """

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html

        return partition_html(filename=self.file_path, **self.unstructured_kwargs)

class HTMLHeaderTextSplitter:
    """
    Splitting HTML files based on specified headers.
    Requires lxml package.
    """
    
    _DEFAULT_HEADER_MAPPING: dict[str, str] = [
        ("h1", "article_main_heading_h1"),
        
        ("h2", "article_subsection_heading_h2"),
        ("h3", "article_subsection_heading_h3"),
        ("h4", "article_subsection_heading_h4"),
        ("h5", "article_subsection_heading_h5"),
        ("h6", "article_subsection_heading_h6"),
        
        ("D2 h1", "sub_article_subsection_heading_h07"),
        ("D2 h2", "sub_article_subsection_heading_h08"),
        ("D2 h3", "sub_article_subsection_heading_h09"),
        ("D2 h4", "sub_article_subsection_heading_h10"),
        ("D2 h5", "sub_article_subsection_heading_h11"),
        ("D2 h6", "sub_article_subsection_heading_h12"),
        
        ("D3 h1", "sub_sub_article_subsection_heading_h13"),
        ("D3 h2", "sub_sub_article_subsection_heading_h14"),
        ("D3 h3", "sub_sub_article_subsection_heading_h15"),
        ("D3 h4", "sub_sub_article_subsection_heading_h16"),
        ("D3 h5", "sub_sub_article_subsection_heading_h17"),
        ("D3 h6", "sub_sub_article_subsection_heading_h18"),
    ]
    
    def __init__(
        self,
        header_mapping: Dict[str, Any] = dict(_DEFAULT_HEADER_MAPPING),
        return_each_element: bool = False,
    ):
        """Create a new HTMLHeaderTextSplitter.
        
        Args:
            header_mapping: dict of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2)].
            return_each_element: Return each element w/ associated headers.
        """

        if header_mapping is None:
            header_mapping = self.DEFAULT_HEADER_MAPPING
        
        self.header_mapping: dict[str, str] = header_mapping
        self.header_capture: Collection[str] = {x[-2:] for x in header_mapping if x[-2:] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']}
        # Output element-by-element or aggregated into chunks w/ common headers
        self.return_each_element: bool = return_each_element
        
        from langchain.document_loaders.parsers.html.html_chunker import HtmlChunker
        self.chunker: HtmlChunker = HtmlChunker(self.header_capture)
    
    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string
        
        Args:
            text: HTML text
        """
        
        return self.split_text_from_source(StringIO(text))

    def split_text_from_source(self, source: any) -> list[Document]:
        """Split HTML file
        
        Args:
            source: either: string (URL or file) or readable IO object (file/connection)
        """
        
        return list(self._docsFromChunks(self.chunker.parseQueue(source, False)))
    def split_text_from_sources(self, sources: Iterable[any]) -> Generator[Document]:
        """Split HTML file
        
        Args:
            sources: a sequence of either: string (URL or file) or readable IO object (file/connection)
        """
        
        for source in sources:
            yield from self._docsFromChunks(self.chunker.parseQueue(source, False))
    
    # Helper Functions
    
    def _docsFromChunks(self, chunks: Collection[dict[str, any]]) -> Generator[Document]:
        for chunk in self._aggregate_chunks_by_metadata(chunks):
            yield self._docFromChunk(chunk)
    def _docFromChunk(self, chunk: dict[str, any]) -> Document:
        return Document(
            page_content=chunk["text"],
            metadata={self.header_mapping.get(key, key): val for key, val in chunk["meta"].items()}
        )
    def _aggregate_chunks_by_metadata(
        self, chunks: Collection[dict[str, any]]
    ) -> Generator[dict[str, any]]:
        """Combine adjacent chunks with identical metadata.
        Should only be called with a single document's chunks.
        
        Args:
            chunks: HTML element content with associated identifying info and metadata
        """
        if self.return_each_element:
            yield from chunks
        else:
            prior_chunk: dict = None
    
            for chunk in chunks:
                if prior_chunk and prior_chunk["meta"] == chunk["meta"]:
                    # If the last chunk in the aggregated list
                    # has the same metadata as the current chunk,
                    # append the current text to the last chunk's text
                    prior_chunk["text"] += "\n  " + chunk["text"]
                else:
                    # Otherwise, yield the prior chunk, and store this new one
                    if prior_chunk:
                        yield prior_chunk
                    prior_chunk = chunk.copy()  # copy to avoid modifying original chunk text
            if prior_chunk:
                yield prior_chunk