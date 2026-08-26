from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader

def load_text_files(directory_path: str, glob_pattern: str, encoding: str = "utf-8"):
    """
    Specifically loads text files from a directory.

    Args:
        directory_path (str): The path to the directory.
        glob_pattern (str): The text file pattern.
        encoding (str): The file encoding.

    Returns:
        list: A list of documents.
    """
    return load_documents_from_directory(
        directory_path,
        glob_pattern,
        TextLoader,
        {'encoding': encoding}
    )

def load_documents_from_directory(directory_path: str, glob_pattern: str, loader_class, loader_kwargs: dict = None):
    """
        Loads documents from a directory using a specific loader.

    Args:
        directory_path (str): The path to the directory to scan.
        glob_pattern (str): The file pattern to include (e.g., "**/*.pdf").
        loader_class: The LangChain loader class (e.g., TextLoader, PyMuPDFLoader).
        loader_kwargs (dict, optional): Additional arguments for the loader.

    Returns:
        list: A list of LangChain documents.
    """
    if loader_kwargs is None:
        loader_kwargs = {}

    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=loader_class,
        loader_kwargs=loader_kwargs,
        show_progress=False
    )
    return loader.load()

def load_pdf_files(directory_path: str, glob_pattern: str):
    """
    Specifically loads PDF files from a directory.

    Args:
        directory_path (str): The path to the directory.
        glob_pattern (str): The PDF file pattern.

    Returns:
        list: A list of documents.
    """
    return load_documents_from_directory(
        directory_path,
        glob_pattern,
        PyMuPDFLoader,
    )