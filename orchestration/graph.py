from langgraph.graph import StateGraph, START, END

from orchestration.state import EmailWorkflowState
from orchestration.nodes import (
    gmail_intake_node,
    classify_email_node,
    finalize_routing_node,
    process_specialist_agent_node,
    complete_agent_node,
)
from orchestration.routing import route_email
from orchestration.checkpointer import get_checkpointer


def _after_intake(state):
    return "classify" if state.get("processing_status") == "email_fetched" else "stop"


def _after_classification(state):
    category = str(state.get("category", "")).upper()
    if category in {"INVOICE", "APPLICATION"}:
        return "agent"
    return "finalize"


def build_email_graph():
    workflow = StateGraph(EmailWorkflowState)

    workflow.add_node("gmail_intake", gmail_intake_node)
    workflow.add_node("classify_email", classify_email_node)
    workflow.add_node("finalize_routing", finalize_routing_node)
    workflow.add_node("process_specialist_agent", process_specialist_agent_node)
    workflow.add_node("complete_agent", complete_agent_node)

    workflow.add_edge(START, "gmail_intake")
    workflow.add_conditional_edges(
        "gmail_intake",
        _after_intake,
        {"classify": "classify_email", "stop": END},
    )
    workflow.add_conditional_edges(
        "classify_email",
        _after_classification,
        {
            "agent": "process_specialist_agent",
            "finalize": "finalize_routing",
        },
    )
    workflow.add_edge("process_specialist_agent", "complete_agent")
    workflow.add_edge("complete_agent", END)
    workflow.add_edge("finalize_routing", END)

    return workflow.compile(checkpointer=get_checkpointer())


email_graph = build_email_graph()
