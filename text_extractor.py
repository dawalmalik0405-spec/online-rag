import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from txt, md, pdf, and docx files with safe encoding handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension in [".txt", ".md", ".markdown"]:
        # Try utf-8 first, fallback to latin-1/cp1252 if utf-8 fails
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(file_path, encoding="latin-1")
            docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    else:
        raise ValueError(f"Unsupported file format '{file_extension}'. Supported: .txt, .md, .pdf, .docx")

# l = extract_text_from_file(file_path)
# print(l)
