import httpx
from typing import List

# Local Ollama configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EXPECTED_DIMENSIONS = 768
TIMEOUT_SECONDS = 120


def generate_embedding(text: str) -> List[float]:
    """
    Generate a 768-dimensional embedding using local Ollama.

    The same function will be used for:
    - Complete resumes
    - Resume sections
    - Job descriptions
    - Individual job requirements
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text.strip()
    }

    try:
        with httpx.Client(
            timeout=TIMEOUT_SECONDS,
            trust_env=False
        ) as client:

            response = client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json=payload
            )

            response.raise_for_status()
            result = response.json()

        embeddings = result.get("embeddings")

        if not embeddings or not embeddings[0]:
            raise ValueError("Ollama returned an empty embedding.")

        embedding = embeddings[0]

        if len(embedding) != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Expected {EXPECTED_DIMENSIONS} dimensions, "
                f"but received {len(embedding)}."
            )

        return embedding

    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to generate embedding using Ollama: {exc}"
        ) from exc
