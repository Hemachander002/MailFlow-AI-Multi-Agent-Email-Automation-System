
from typing import Literal

from orchestration.state import EmailWorkflowState


Route = Literal[
    "invoice",
    "application",
    "skip",
    "review",
]


def route_email(state: EmailWorkflowState) -> Route:
    """
    Choose the next workflow branch based on email classification.
    """

    category = state.get("category", "").strip().upper()

    if category == "INVOICE":
        return "invoice"

    if category == "APPLICATION":
        return "application"

    if category in {"SPAM", "OTHER"}:
        return "skip"

    if category == "REVIEW":
        return "review"

    # Unknown or missing categories must never trigger an agent.
    return "review"
