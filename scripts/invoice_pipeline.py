
import os
import json
import hashlib
import mimetypes
from pathlib import Path

import httpx
import psycopg
import pytesseract
from dotenv import dotenv_values
from PIL import Image
from app_observability import observe_api_call, observe_agent, record_invoice_result


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV = dotenv_values(BASE_DIR / ".env")

DATABASE_URL = ENV.get("DATABASE_URL")
OLLAMA_BASE_URL = ENV.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_MODEL = ENV.get("OLLAMA_MODEL", "invoice-extractor")

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

IMAGE_PATH = (
    BASE_DIR
    / "data"
    / "invoices"
    / "images"
    / "batch1-0001.jpg"
)

OUTPUT_DIR = BASE_DIR / "data" / "invoices" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# STEP 1: OCR
# ==================================================

def extract_text(image_path: Path) -> str:
    print("Running Tesseract OCR...")

    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )

    if not text.strip():
        raise ValueError("OCR returned empty text.")

    print("OCR completed.")
    return text


# ==================================================
# STEP 2: OLLAMA INVOICE EXTRACTION
# ==================================================

def extract_invoice_data(ocr_text: str) -> dict:
    print("Sending OCR text to Ollama...")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract invoice information from OCR text. "
                    "Return only valid JSON. Follow the invoice schema "
                    "and preserve the missing-value conventions used "
                    "in the training data. Do not invent values."
                )
            },
            {
                "role": "user",
                "content": ocr_text
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 700
        }
    }

    with observe_api_call("ollama", "invoice_extraction"):
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300.0,
            trust_env=False,
        )
    response.raise_for_status()

    model_output = response.json()["message"]["content"]

    try:
        invoice_data = json.loads(model_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON."
        ) from exc

    if not isinstance(invoice_data, dict):
        raise ValueError("Invoice output must be a JSON object.")

    print("Invoice extraction completed.")
    return invoice_data


# ==================================================
# STEP 3: PREPARE DATABASE VALUES
# ==================================================


from datetime import datetime

def prepare_invoice_values(invoice_data: dict) -> dict:
    invoice = invoice_data.get("invoice", {})
    subtotal = invoice_data.get("subtotal", {})
    items = invoice_data.get("items", [])

    if not isinstance(invoice, dict):
        raise ValueError("'invoice' must be a JSON object.")

    if not isinstance(subtotal, dict):
        raise ValueError("'subtotal' must be a JSON object.")

    if not isinstance(items, list):
        raise ValueError("'items' must be a JSON list.")

    def get_text(source, key):
        value = source.get(key)

        if value is None or value == "":
            return None

        if isinstance(value, (dict, list)):
            raise ValueError(f"Invalid scalar field: {key}")

        return str(value)

    # Convert invoice date into a PostgreSQL DATE value.
    invoice_date = get_text(invoice, "invoice_date")

    parsed_date = None

    if invoice_date:
        try:
            parsed_date = datetime.strptime(
                invoice_date, "%m/%d/%Y"
            ).date()
        except ValueError:
            raise ValueError(
                f"Unsupported invoice date format: {invoice_date}"
            )

    return {
        "client_name": get_text(invoice, "client_name"),
        "seller_name": get_text(invoice, "seller_name"),
        "invoice_number": get_text(invoice, "invoice_number"),
        "date": parsed_date,
        "tax": get_text(subtotal, "tax"),
        "total": get_text(subtotal, "total"),
        "invoice_data": json.dumps(
            invoice_data, ensure_ascii=False
        ),
        "items_count": len(items)
    }



# ==================================================
# STEP 4: SAVE INVOICE AND IMAGE IN ONE TRANSACTION
# ==================================================


def save_invoice_and_image(
    invoice_data: dict,
    image_path: Path
):
    print("Saving invoice and original image to PostgreSQL...")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing from .env")

    values = prepare_invoice_values(invoice_data)

    invoice_number = values["invoice_number"]

    image_bytes = image_path.read_bytes()

    mime_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "application/octet-stream"
    )

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    invoice_sql = """
        INSERT INTO invoices (
            client_name,
            seller_name,
            tax,
            total,
            invoice_number,
            invoice_data,
            "date",
            items_count
        )
        VALUES (
            %(client_name)s,
            %(seller_name)s,
            %(tax)s,
            %(total)s,
            %(invoice_number)s,
            %(invoice_data)s::jsonb,
            %(date)s,
            %(items_count)s
        )
        RETURNING id;
    """

    image_sql = """
        INSERT INTO invoice_images (
            invoice_id,
            file_name,
            mime_type,
            image_data,
            sha256,
            invoice_number
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    db_url = DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1
    )

    try:
        # Both inserts commit together or roll back together.
        with psycopg.connect(db_url) as connection:
            with connection.cursor() as cursor:

                # Check for an existing invoice number.
                # Skip the check when the invoice number is missing.
                if invoice_number:
                    cursor.execute(
                        """
                        SELECT invoice_id
                        FROM invoice_images
                        WHERE invoice_number = %s
                        LIMIT 1;
                        """,
                        (invoice_number,)
                    )

                    existing_invoice = cursor.fetchone()

                    if existing_invoice:
                        print("Duplicate invoice detected. Skipping.")

                        return {
                            "status": "duplicate",
                            "invoice_id": existing_invoice[0]
                        }

                # Insert invoice.
                cursor.execute(invoice_sql, values)

                invoice_id = cursor.fetchone()[0]

                # Insert original image and invoice number.
                cursor.execute(
                    image_sql,
                    (
                        invoice_id,
                        image_path.name,
                        mime_type,
                        psycopg.Binary(image_bytes),
                        image_hash,
                        invoice_number
                    )
                )

        print("Database transaction committed.")

        return {
            "status": "inserted",
            "invoice_id": invoice_id
        }

    except psycopg.errors.UniqueViolation as exc:

        # Handles a duplicate detected by the database constraint,
        # including a race between simultaneous processing attempts.
        if (
            exc.diag.constraint_name
            == "uq_invoice_images_invoice_number"
        ):
            print("Duplicate invoice detected by database. Skipping.")

            return {
                "status": "duplicate",
                "invoice_id": None
            }

        raise

def _process_invoice_impl(image_path):
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Invoice image not found: {image_path}")

    print("\n========== INVOICE PIPELINE ==========\n")
    ocr_text = extract_text(image_path)
    invoice_data = extract_invoice_data(ocr_text)
    save_result = save_invoice_and_image(invoice_data, image_path)

    if save_result["status"] == "duplicate":
        print("\n========== DUPLICATE SKIPPED ==========")
        return {
            "status": "duplicate",
            "invoice_id": save_result.get("invoice_id"),
        }

    invoice_id = save_result["invoice_id"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = OUTPUT_DIR / f"{image_path.stem}_result.json"
    result = {
        "invoice_id": str(invoice_id),
        "source_image": image_path.name,
        "invoice_data": invoice_data,
    }
    result_file.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Invoice processing completed successfully.")
    return {
        "status": "inserted",
        "invoice_id": str(invoice_id),
        "result_file": str(result_file),
    }


def process_invoice(image_path):
    """Instrument the existing invoice pipeline without changing its logic."""
    with observe_agent("invoice"):
        result = _process_invoice_impl(image_path)
        record_invoice_result(result)
        return result

# ==================================================
# MAIN PIPELINE
# ==================================================


def main():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Invoice image not found: {IMAGE_PATH}"
        )

    print("\n========== INVOICE PIPELINE ==========\n")

    # OCR
    ocr_text = extract_text(IMAGE_PATH)

    # Local AI extraction
    invoice_data = extract_invoice_data(ocr_text)

    # Save invoice + original image
    save_result = save_invoice_and_image(
        invoice_data,
        IMAGE_PATH
    )

    # Stop processing if this invoice is a duplicate.
    if save_result["status"] == "duplicate":
        print("\n========== DUPLICATE SKIPPED ==========")
        print("This invoice number is already registered.")
        print("No new invoice or image was saved.")
        print("=======================================\n")
        return

    invoice_id = save_result["invoice_id"]

    # Save a local JSON copy for inspection
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_file = OUTPUT_DIR / (
        f"{IMAGE_PATH.stem}_result.json"
    )

    result = {
        "invoice_id": str(invoice_id),
        "source_image": IMAGE_PATH.name,
        "invoice_data": invoice_data
    }

    result_file.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== SUCCESS ==========")
    print("Invoice record and image saved successfully.")
    print("=============================\n")


if __name__ == "__main__":
    main()

