from io import StringIO
from typing import (
    List,
    Dict,
    Iterable, 
    Any,
    Iterator,
    Collection,
    Generator,
    Sequence,
    Optional,
    Literal,
)

from langchain.schema import Document, BaseDocumentTransformer
from langchain.document_loaders.base import BaseLoader

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



class HeaderChunkedHTMLLoader(BaseLoader):
    """
    Splitting HTML files based on specified headers.
    Requires lxml package.
    """

    # This default header mapping "flattens" metadata when it finds higher-level headers "deeper" than lower-level headers
    # before them, i.e. a higher-level header (e.g. h1) is "deeper" (e.g. "D2") when it's nested in an element with a
    # lower-level header (e.g. h4).
    _DEFAULT_HEADER_MAPPING: Dict[str, str] = {
        "h1": "article_main_heading_h1",
    
        "h2": "article_subsection_heading_h2",
        "h3": "article_subsection_heading_h3",
        "h4": "article_subsection_heading_h4",
        "h5": "article_subsection_heading_h5",
        "h6": "article_subsection_heading_h6",
    
        "D2 h1": "sub_article_subsection_heading_h07",
        "D2 h2": "sub_article_subsection_heading_h08",
        "D2 h3": "sub_article_subsection_heading_h09",
        "D2 h4": "sub_article_subsection_heading_h10",
        "D2 h5": "sub_article_subsection_heading_h11",
        "D2 h6": "sub_article_subsection_heading_h12",
    
        "D3 h1": "sub_sub_article_subsection_heading_h13",
        "D3 h2": "sub_sub_article_subsection_heading_h14",
        "D3 h3": "sub_sub_article_subsection_heading_h15",
        "D3 h4": "sub_sub_article_subsection_heading_h16",
        "D3 h5": "sub_sub_article_subsection_heading_h17",
        "D3 h6": "sub_sub_article_subsection_heading_h18",
    }

    def __init__(
        self,
        sources: Union[any, Iterable[any]],
        header_mapping: Dict[str, Any] = None,
        return_each_element: bool = False,
        return_urls: bool = True,
        use_selenium: bool = False,
    ):
        """Create a new HeaderChunkedHTMLLoader.
        
        Args:
            sources: if use_selenium=True, these must be URL strings
            header_mapping: dict of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2)].
                If 'None' (default), _DEFAULT_HEADER_MAPPING is used.
            return_each_element: Return each element w/ associated headers.
                'False' to combine all adjacent chunks with identical metadata.
            return_urls: whether to include a source-url metadata in each returned Document
        """
        from langchain.document_loaders.parsers.html.html_chunker import HtmlChunker


        if not sources:
            raise Exception(f"HeaderChunkedHTMLLoader(sources={sources})")

        self.sources: Iterable[any] = sources if isinstance(sources, Iterable) else [sources]

        self.header_mapping: dict[str, str] = self._DEFAULT_HEADER_MAPPING.copy() \
            if header_mapping is None else header_mapping
        self.header_capture: set[str] = {
            x[-2:] for x in header_mapping if x[-2:] in [
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6']}
        
        self.return_each_element: bool = return_each_element
        self.return_urls: bool = return_urls
        self.use_selenium: bool = use_selenium

        self.chunker: HtmlChunker = HtmlChunker(
            "selenium" if use_selenium else "lxml",
            self.header_capture)

    # TODO remove this when it is implemented in abstract parent class
    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        yield from self._aggregate(
            self._docs_from_chunks(
                self.chunker.parse_chunk_sequence(sources)))

    # Helper Functions

    def _aggregate(self, documents: Iterable[Document]) -> Iterator[Document]:
        if self.return_each_element:
            return iter(documents)
        else:
            return DocumentMetadataCleaver().transform_documents(documents)

    def _docs_from_chunks(self, chunks: Collection[dict[str, any]]) -> Generator[Document, None, None]:
        for chunk in chunks:
            yield self._doc_from_chunk(chunk)

    def _doc_from_chunk(self, chunk: dict[str, any]) -> Document:
        meta = {}
        if self.return_urls:
            meta["url"] = chunk["uri"]
        for key, val in chunk["meta"].items():
            mappedKey = self.header_mapping[key, None]
            meta[key if mappedKey is None else mappedKey] = val
        return Document(
            page_content=chunk["text"],
            metadata=meta
        )


class HeaderChunkedHTMLLoaderFromString(HeaderChunkedHTMLLoader):
    def __init__(
        self,
        sources: Iterable[str],
        header_mapping: Dict[str, Any] = None,
        return_each_element: bool = False,
    ):
        """Args:
            sources: a sequence of html-text strings
        """

        super().__init__(
            (StringIO(source) for source in sources),
            header_mapping,
            return_each_element,
            False,
            False)


class DocumentMetadataCleaver(BaseDocumentTransformer):
    """
    This "reverse TextSplitter" combines adjacent documents if they have the same metadata.
    """

    # XXX TODO 2023-11-08 n.b. the parent abstract method signature is :Sequence->Sequence
    # that seems incorrect/inefficient, especially the return-type.
    # does a list-comprehension resolve this efficiently?
    # their example shows: return [stateful_documents[i] for i in sorted(included_idxs)]
    def transform_documents(
        self, documents: Iterable[Document], **kwargs: Any
    ) -> Iterator[Document]:

        prior_doc: Optional[Document] = None
        for doc in documents:
            if prior_doc and prior_doc.metadata == doc.metadata:
                # If the last chunk in the aggregated list
                # has the same metadata as the current chunk,
                # append the current text to the last chunk's text
                prior_doc.page_content += "\n\n" + doc.page_content
            else:
                # Otherwise, yield the prior chunk, and store this new one
                if prior_doc:
                    yield prior_doc
                prior_doc = Document(  # copy to avoid modifying original chunk text
                    metadata=doc.metadata,
                    page_content=doc.page_content
                )
        # send the final chunk, when done looping
        if prior_doc:
            yield prior_doc
