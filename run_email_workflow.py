
import logging
import time
import uuid

from orchestration.graph import email_graph
from app_observability import (
    start_metrics_server, record_email_processed, record_workflow_error, trace_span,
)


POLL_INTERVAL_SECONDS = 5
MAX_EMAILS_PER_CYCLE = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def process_available_emails():
    """Process eligible emails one at a time until the inbox is drained."""
    processed = 0

    while processed < MAX_EMAILS_PER_CYCLE:
        thread_id = f"email-worker-{uuid.uuid4().hex}"

        try:
            with trace_span("workflow.email_execution"):
                result = email_graph.invoke(
                    {"current_step": "started"},
                    config={
                        "configurable": {
                            "thread_id": thread_id
                        }
                    },
                )
        except Exception:
            record_email_processed("UNKNOWN", "FAILED")
            raise

        status = result.get("processing_status", "unknown")

        if status in {
            "no_eligible_emails",
            "no_unprocessed_emails",
        }:
            break

        processed += 1
        record_email_processed(result.get("category", "UNKNOWN"), status)

        # Do not log message IDs, email addresses, subjects, or attachment names.
        logging.info(
            "Email workflow item completed | category=%s | status=%s | step=%s",
            result.get("category", "unknown"),
            status,
            result.get("current_step", "unknown"),
        )

    return processed


def main():
    start_metrics_server()
    logging.info("Multi-Agent Email Workflow started.")
    logging.info("Prometheus metrics endpoint listening on port 8000.")
    logging.info("Polling interval: %s seconds", POLL_INTERVAL_SECONDS)

    while True:
        try:
            count = process_available_emails()

            if count == 0:
                logging.info("No unprocessed eligible emails found.")

        except KeyboardInterrupt:
            logging.info("Workflow stopped by user.")
            break

        except Exception as error:
            record_workflow_error(error)
            logging.error(
                "Workflow execution failed. Error type: %s",
                type(error).__name__,
            )

        logging.info("Next inbox check in %s seconds.", POLL_INTERVAL_SECONDS)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
