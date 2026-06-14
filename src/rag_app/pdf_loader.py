from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(file_path))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text
