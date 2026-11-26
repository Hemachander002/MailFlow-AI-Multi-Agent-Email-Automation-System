
from typing import TypedDict, Any


class EmailWorkflowState(TypedDict, total=False):
    """
    Shared state for processing one Gmail message
    through the LangGraph workflow.
    """

    # Gmail message information
    message_id: str
    thread_id: str
    internal_date_ms: int
    received_at_ist: str

    # Minimal email information
    sender_display_name: str
    sender_domain: str
    subject: str
    body_excerpt: str

    # Attachment metadata and downloaded files
    attachment_metadata: list[dict[str, Any]]
    local_attachment_paths: list[str]

    # Gemini classification results
    category: str
    confidence: float
    classification_reason: str
    needs_review: bool

    # Google Drive upload results
    drive_file_ids: list[str]
    drive_file_links: list[str]

    # Specialist agent information
    job_id: str
    agent_result: dict[str, Any]

    # Workflow tracking
    current_step: str
    processing_status: str
    error: str
    error_code: str
    retryable: bool
