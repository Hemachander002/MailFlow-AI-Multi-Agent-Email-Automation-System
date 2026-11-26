
from pathlib import Path
import fitz
import pytesseract
from PIL import Image
import io


# Project and output directories
BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = BASE_DIR / "data" / "resumes" / "candidate"
OUTPUT_TEXT_PATH = CANDIDATE_DIR / "candidate_resume.txt"

# Windows Tesseract installation
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_resume_text(pdf_path: str) -> str:
    """
    Extract text from a PDF resume.
    First attempts normal text extraction.
    If the PDF is scanned, falls back to Tesseract OCR.

    Saves extracted text in the candidate folder.
    """

    file_path = Path(pdf_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Resume PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF resumes are supported.")

    extracted_pages = []

    with fitz.open(file_path) as document:

        if document.is_encrypted:
            raise ValueError(
                "Encrypted PDF resumes are not supported."
            )

        # First attempt: normal PDF text extraction.
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append(
                    f"\n--- Page {page_number} ---\n{page_text}"
                )

        # Second attempt: OCR if no selectable text exists.
        if not extracted_pages:

            print("Scanned PDF detected. Starting OCR...")

            for page_number, page in enumerate(document, start=1):

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                page_text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--oem 3 --psm 6"
                ).strip()

                if page_text:
                    extracted_pages.append(
                        f"\n--- Page {page_number} ---\n{page_text}"
                    )

    resume_text = "\n".join(extracted_pages).strip()

    if not resume_text:
        raise ValueError(
            "No readable text found in the PDF, even after OCR."
        )

    # Save original extracted resume text.
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_TEXT_PATH.write_text(
        resume_text,
        encoding="utf-8"
    )

    print("Resume text extraction successful.")
    print(f"Extracted text saved to: {OUTPUT_TEXT_PATH}")

    return resume_text

