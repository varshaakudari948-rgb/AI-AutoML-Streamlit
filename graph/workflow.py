import os

from typing import Literal

from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    InMemorySaver
)

from langgraph.types import interrupt

from graph.state import WorkflowState

from graph.nodes import (
    data_preparation_node,
    feature_engineering_node,
    ml_experiment_node,
    explainability_node,
    database_node,
    report_generation_node
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RETRAINS = int(
    os.getenv(
        "MAX_RETRAINS",
        "2"
    )
)

MIN_SCORE = float(
    os.getenv(
        "MIN_ACCEPTABLE_SCORE",
        "0.80"
    )
)


# ============================================================
# START EXPERIMENT NODE
# ============================================================

def start_experiment(
    state: WorkflowState
) -> dict:

    print("\n" + "=" * 60)
    print("AUTOXLAB EXPERIMENT STARTED")
    print("=" * 60)

    return {

        "experiment_number":
            state.get(
                "experiment_number",
                1
            ),

        "retrain_count":
            state.get(
                "retrain_count",
                0
            ),

        "max_retrains":
            state.get(
                "max_retrains",
                MAX_RETRAINS
            ),

        "minimum_score":
            state.get(
                "minimum_score",
                MIN_SCORE
            ),

        "status":
            "experiment_started"
    }


# ============================================================
# CONDITIONAL EDGE
# ============================================================

def decide_retraining(
    state: WorkflowState
) -> Literal[
    "retrain",
    "continue"
]:

    score = state.get(
        "best_score",
        0.0
    )

    retrain_count = state.get(
        "retrain_count",
        0
    )

    max_retrains = state.get(
        "max_retrains",
        MAX_RETRAINS
    )

    minimum_score = state.get(
        "minimum_score",
        MIN_SCORE
    )

    print(
        f"\n[AutoXLab] Model score: {score:.4f}"
    )

    print(
        f"[AutoXLab] Required score: "
        f"{minimum_score:.4f}"
    )

    print(
        f"[AutoXLab] Retrain count: "
        f"{retrain_count}/{max_retrains}"
    )

    # --------------------------------------------------------
    # SCORE ACCEPTABLE
    # --------------------------------------------------------

    if score >= minimum_score:

        print(
            "[AutoXLab] Score acceptable."
        )

        return "continue"

    # --------------------------------------------------------
    # RETRAIN
    # --------------------------------------------------------

    if retrain_count < max_retrains:

        print(
            "[AutoXLab] Score below threshold."
        )

        print(
            "[AutoXLab] Starting retraining."
        )

        return "retrain"

    # --------------------------------------------------------
    # MAX RETRAINING REACHED
    # --------------------------------------------------------

    print(
        "[AutoXLab] Maximum retraining reached."
    )

    return "continue"


# ============================================================
# RETRAIN NODE
# ============================================================

def prepare_retraining(
    state: WorkflowState
) -> dict:

    current_retrain_count = state.get(
        "retrain_count",
        0
    )

    current_experiment = state.get(
        "experiment_number",
        1
    )

    return {

        "retrain_count":
            current_retrain_count + 1,

        "experiment_number":
            current_experiment + 1,

        "status":
            "retraining_started"
    }


# ============================================================
# HUMAN APPROVAL NODE
# ============================================================

def human_model_approval(
    state: WorkflowState
) -> dict:

    approval = interrupt({

        "type":
            "model_approval",

        "message":
            "The autonomous pipeline has "
            "finished model selection.",

        "best_model":
            state.get(
                "best_model_name",
                "Unknown"
            ),

        "score":
            state.get(
                "best_score",
                0.0
            ),

        "metrics":
            state.get(
                "metrics",
                {}
            ),

        "question":
            "Approve this model?"
    })

    # --------------------------------------------------------
    # CONVERT APPROVAL TO BOOLEAN
    # --------------------------------------------------------

    approved = False

    if isinstance(
        approval,
        dict
    ):

        approved = bool(
            approval.get(
                "approved",
                False
            )
        )

    else:

        approved = bool(
            approval
        )

    # --------------------------------------------------------
    # APPROVAL MESSAGE
    # --------------------------------------------------------

    if approved:

        message = (
            "Human approved the final model."
        )

    else:

        message = (
            "Human rejected the final model."
        )

    print(
        f"\n[AutoXLab] {message}"
    )

    return {

        "human_approved":
            approved,

        "approval_message":
            message,

        "status":
            "human_approval_completed"
    }


# ============================================================
# FINALIZE NODE
# ============================================================

def finalize_experiment(
    state: WorkflowState
) -> dict:

    print("\n" + "=" * 60)
    print("AUTOXLAB EXPERIMENT FINALIZED")
    print("=" * 60)

    approved = state.get(
        "human_approved",
        False
    )

    if approved:

        status = (
            "Experiment finalized. "
            "Final model approved."
        )

    else:

        status = (
            "Experiment finalized. "
            "Final model was not approved."
        )

    return {

        "status":
            status,

        "messages": [
            status
        ]
    }


# ============================================================
# CREATE STATE GRAPH
# ============================================================

builder = StateGraph(
    WorkflowState
)


# ============================================================
# ADD NODES
# ============================================================

builder.add_node(
    "start_experiment",
    start_experiment
)

builder.add_node(
    "data_preparation",
    data_preparation_node
)

builder.add_node(
    "feature_engineering",
    feature_engineering_node
)

builder.add_node(
    "ml_experiment",
    ml_experiment_node
)

builder.add_node(
    "prepare_retraining",
    prepare_retraining
)

builder.add_node(
    "explainability",
    explainability_node
)

builder.add_node(
    "database",
    database_node
)

builder.add_node(
    "report_generation",
    report_generation_node
)

builder.add_node(
    "human_approval",
    human_model_approval
)

builder.add_node(
    "finalize",
    finalize_experiment
)


# ============================================================
# START
# ============================================================

builder.add_edge(
    START,
    "start_experiment"
)


# ============================================================
# AGENT-TO-AGENT CONNECTIONS
# ============================================================

builder.add_edge(
    "start_experiment",
    "data_preparation"
)

builder.add_edge(
    "data_preparation",
    "feature_engineering"
)

builder.add_edge(
    "feature_engineering",
    "ml_experiment"
)


# ============================================================
# CONDITIONAL EDGE
# ============================================================

builder.add_conditional_edges(

    "ml_experiment",

    decide_retraining,

    {

        "retrain":
            "prepare_retraining",

        "continue":
            "explainability"
    }
)


# ============================================================
# RETRAIN LOOP
# ============================================================

builder.add_edge(

    "prepare_retraining",

    "ml_experiment"
)


# ============================================================
# EXPLAINABILITY → DATABASE
# ============================================================

builder.add_edge(

    "explainability",

    "database"
)


# ============================================================
# DATABASE → REPORT
# ============================================================

builder.add_edge(

    "database",

    "report_generation"
)


# ============================================================
# REPORT → HUMAN APPROVAL
# ============================================================

builder.add_edge(

    "report_generation",

    "human_approval"
)


# ============================================================
# HUMAN APPROVAL → FINALIZE
# ============================================================

builder.add_edge(

    "human_approval",

    "finalize"
)


# ============================================================
# FINALIZE → END
# ============================================================

builder.add_edge(

    "finalize",

    END
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# COMPILE GRAPH
# ============================================================

graph = builder.compile(
    checkpointer=checkpointer
)


print(
    "\n[AutoXLab] LangGraph workflow compiled successfully."
)