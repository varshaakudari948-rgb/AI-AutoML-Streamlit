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

    experiment_number = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    max_retrains = int(
        state.get(
            "max_retrains",
            MAX_RETRAINS
        ) or MAX_RETRAINS
    )

    minimum_score = float(
        state.get(
            "minimum_score",
            MIN_SCORE
        ) or MIN_SCORE
    )

    print(
        f"Experiment Number: "
        f"{experiment_number}"
    )

    print(
        f"Retrain Count: "
        f"{retrain_count}/{max_retrains}"
    )

    print(
        f"Minimum Score: "
        f"{minimum_score:.4f}"
    )

    return {

        "experiment_number":
            experiment_number,

        "retrain_count":
            retrain_count,

        "max_retrains":
            max_retrains,

        "minimum_score":
            minimum_score,

        "status":
            "experiment_started"
    }


# ============================================================
# AUTOMATIC RETRAIN DECISION
# ============================================================

def decide_retraining(
    state: WorkflowState
) -> Literal[
    "retrain",
    "continue"
]:

    score = float(
        state.get(
            "best_score",
            0.0
        ) or 0.0
    )

    retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    max_retrains = int(
        state.get(
            "max_retrains",
            MAX_RETRAINS
        ) or MAX_RETRAINS
    )

    minimum_score = float(
        state.get(
            "minimum_score",
            MIN_SCORE
        ) or MIN_SCORE
    )

    experiment_number = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    print("\n" + "-" * 60)

    print(
        f"[AutoXLab] Experiment: "
        f"{experiment_number}"
    )

    print(
        f"[AutoXLab] Model score: "
        f"{score:.4f}"
    )

    print(
        f"[AutoXLab] Required score: "
        f"{minimum_score:.4f}"
    )

    print(
        f"[AutoXLab] Retrain count: "
        f"{retrain_count}/{max_retrains}"
    )

    # ========================================================
    # SCORE IS ACCEPTABLE
    # ========================================================

    if score >= minimum_score:

        print(
            "[AutoXLab] Score acceptable."
        )

        print(
            "[AutoXLab] Continuing to explainability."
        )

        return "continue"

    # ========================================================
    # SCORE IS LOW AND RETRAINING IS AVAILABLE
    # ========================================================

    if retrain_count < max_retrains:

        print(
            "[AutoXLab] Score below threshold."
        )

        print(
            "[AutoXLab] Automatic retraining required."
        )

        return "retrain"

    # ========================================================
    # MAXIMUM RETRAINING REACHED
    # ========================================================

    print(
        "[AutoXLab] Maximum retraining reached."
    )

    print(
        "[AutoXLab] Continuing with current best model."
    )

    return "continue"


# ============================================================
# PREPARE RETRAINING
# ============================================================

def prepare_retraining(
    state: WorkflowState
) -> dict:

    current_retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    current_experiment = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    max_retrains = int(
        state.get(
            "max_retrains",
            MAX_RETRAINS
        ) or MAX_RETRAINS
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if current_retrain_count >= max_retrains:

        print("\n" + "=" * 60)
        print("MAXIMUM RETRAINING LIMIT REACHED")
        print("=" * 60)

        return {

            "status":
                "max_retraining_reached",

            "retrain_required":
                False,

            "approval_status":
                "rejected"
        }

    # ========================================================
    # INCREMENT COUNTERS
    # ========================================================

    next_retrain_count = (
        current_retrain_count + 1
    )

    next_experiment = (
        current_experiment + 1
    )

    print("\n" + "=" * 60)
    print("AUTOXLAB - RETRAINING STARTED")
    print("=" * 60)

    print(
        f"Previous Experiment: "
        f"{current_experiment}"
    )

    print(
        f"New Experiment: "
        f"{next_experiment}"
    )

    print(
        f"Retrain Count: "
        f"{next_retrain_count}/{max_retrains}"
    )

    print(
        "[AutoXLab] Previous model rejected "
        "or score was below threshold."
    )

    print(
        "[AutoXLab] Starting a fresh ML experiment."
    )

    # ========================================================
    # RETURN UPDATED STATE
    # ========================================================

    return {

        # Increment retraining count
        "retrain_count":
            next_retrain_count,

        # Increment experiment number
        "experiment_number":
            next_experiment,

        # Mark retraining
        "retrain_required":
            True,

        "retrain_reason":
            "Previous model was rejected "
            "or score was below threshold.",

        "status":
            "retraining_started",

        # Clear previous analysis
        "llm_analysis":
            "",

        # Clear old approval
        "human_approved":
            False,

        "approval_status":
            "retraining",

        "approval_message":
            "",

        "error":
            ""
    }


# ============================================================
# HUMAN APPROVAL NODE
# ============================================================

def human_model_approval(
    state: WorkflowState
) -> dict:

    experiment_number = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    max_retrains = int(
        state.get(
            "max_retrains",
            MAX_RETRAINS
        ) or MAX_RETRAINS
    )

    best_model_name = state.get(
        "best_model_name",
        "Unknown"
    )

    best_score = float(
        state.get(
            "best_score",
            0.0
        ) or 0.0
    )

    metrics = state.get(
        "metrics",
        {}
    )

    # ========================================================
    # INTERRUPT FOR HUMAN
    # ========================================================

    approval = interrupt({

        "type":
            "model_approval",

        "message":
            (
                "The autonomous pipeline has "
                "finished model selection."
            ),

        "experiment_number":
            experiment_number,

        "retrain_count":
            retrain_count,

        "max_retrains":
            max_retrains,

        "best_model":
            best_model_name,

        "score":
            best_score,

        "metrics":
            metrics,

        "question":
            "Approve this model?"
    })

    # ========================================================
    # CONVERT APPROVAL TO BOOLEAN
    # ========================================================

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

    # ========================================================
    # HANDLE APPROVAL
    # ========================================================

    if approved:

        message = (
            "Human approved the model."
        )

        status = "approved"

        print(
            "\n[AutoXLab] "
            "Human APPROVED model."
        )

    else:

        message = (
            "Human rejected the model."
        )

        status = "rejected"

        print(
            "\n[AutoXLab] "
            "Human REJECTED model."
        )

        print(
            "[AutoXLab] "
            "The workflow will go to retraining."
        )

    # ========================================================
    # RETURN HUMAN DECISION
    # ========================================================

    return {

        "human_approved":
            approved,

        "approval_status":
            status,

        "approval_message":
            message,

        "status":
            "human_approval_completed"
    }


# ============================================================
# ROUTE AFTER HUMAN APPROVAL
# ============================================================

def route_after_human_approval(
    state: WorkflowState
) -> Literal[
    "retrain",
    "finalize"
]:

    approved = bool(
        state.get(
            "human_approved",
            False
        )
    )

    retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    max_retrains = int(
        state.get(
            "max_retrains",
            MAX_RETRAINS
        ) or MAX_RETRAINS
    )

    experiment_number = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    # ========================================================
    # APPROVED
    # ========================================================

    if approved:

        print("\n" + "=" * 60)

        print(
            "[AutoXLab] HUMAN APPROVAL = TRUE"
        )

        print(
            "[AutoXLab] Model approved."
        )

        print(
            "[AutoXLab] Going to finalization."
        )

        print("=" * 60)

        return "finalize"

    # ========================================================
    # REJECTED + RETRAINING AVAILABLE
    # ========================================================

    if retrain_count < max_retrains:

        print("\n" + "=" * 60)

        print(
            "[AutoXLab] HUMAN APPROVAL = FALSE"
        )

        print(
            "[AutoXLab] Model rejected."
        )

        print(
            f"[AutoXLab] Experiment "
            f"{experiment_number} will be retrained."
        )

        print(
            f"[AutoXLab] Retraining "
            f"{retrain_count + 1}/{max_retrains}"
        )

        print(
            "[AutoXLab] Routing to prepare_retraining."
        )

        print("=" * 60)

        return "retrain"

    # ========================================================
    # REJECTED + NO RETRAINING LEFT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "[AutoXLab] Model rejected."
    )

    print(
        "[AutoXLab] Maximum retraining limit reached."
    )

    print(
        "[AutoXLab] Finalizing without approval."
    )

    print("=" * 60)

    return "finalize"


# ============================================================
# FINALIZE EXPERIMENT
# ============================================================

def finalize_experiment(
    state: WorkflowState
) -> dict:

    print("\n" + "=" * 60)
    print("AUTOXLAB EXPERIMENT FINALIZED")
    print("=" * 60)

    approved = bool(
        state.get(
            "human_approved",
            False
        )
    )

    experiment_number = int(
        state.get(
            "experiment_number",
            1
        ) or 1
    )

    retrain_count = int(
        state.get(
            "retrain_count",
            0
        ) or 0
    )

    best_model_name = state.get(
        "best_model_name",
        "Unknown"
    )

    best_score = float(
        state.get(
            "best_score",
            0.0
        ) or 0.0
    )

    # ========================================================
    # APPROVED
    # ========================================================

    if approved:

        status = (
            f"Experiment {experiment_number} finalized. "
            f"Final model approved."
        )

        print(
            f"[AutoXLab] Final model: "
            f"{best_model_name}"
        )

        print(
            f"[AutoXLab] Final score: "
            f"{best_score:.4f}"
        )

        print(
            "[AutoXLab] Human approved the model."
        )

    # ========================================================
    # NOT APPROVED
    # ========================================================

    else:

        status = (
            f"Experiment {experiment_number} finalized. "
            f"Final model was not approved."
        )

        print(
            "[AutoXLab] Final model was not approved."
        )

    print(
        f"[AutoXLab] Total retraining loops: "
        f"{retrain_count}"
    )

    print(
        "=" * 60
    )

    return {

        "status":
            status,

        "messages": [

            status,

            (
                f"Completed experiment: "
                f"{experiment_number}"
            ),

            (
                f"Total retraining loops: "
                f"{retrain_count}"
            ),

            (
                f"Best model: "
                f"{best_model_name}"
            ),

            (
                f"Best score: "
                f"{best_score:.4f}"
            )
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
# NORMAL PIPELINE
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
# ML EXPERIMENT → AUTOMATIC RETRAIN DECISION
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
# RETRAINING LOOP
#
# prepare_retraining
#       ↓
# ML Experiment
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
# HUMAN APPROVAL → CONDITIONAL ROUTING
#
# APPROVE
#     ↓
# FINALIZE
#
# REJECT
#     ↓
# PREPARE RETRAINING
#     ↓
# ML EXPERIMENT
# ============================================================

builder.add_conditional_edges(

    "human_approval",

    route_after_human_approval,

    {

        "retrain":
            "prepare_retraining",

        "finalize":
            "finalize"
    }
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


# ============================================================
# CONFIRMATION
# ============================================================

print(
    "\n[AutoXLab] "
    "LangGraph workflow compiled successfully."
)