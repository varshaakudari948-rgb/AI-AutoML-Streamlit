from typing import Any

from graph.state import WorkflowState

from agents.data_preparation.agent import DataPreparationAgent
from agents.feature_engineering.agent import FeatureEngineeringAgent
from agents.ml_experiment.agent import MLExperimentAgent
from agents.explainability.agent import ExplainabilityAgent
from agents.database.agent import DatabaseAgent
from agents.report_generation.agent import ReportGenerationAgent


# ============================================================
# CREATE AGENTS
# ============================================================

data_agent = DataPreparationAgent()
feature_agent = FeatureEngineeringAgent()
ml_agent = MLExperimentAgent()
explainability_agent = ExplainabilityAgent()
database_agent = DatabaseAgent()
report_agent = ReportGenerationAgent()


# ============================================================
# DATA PREPARATION NODE
# ============================================================

def data_preparation_node(state: WorkflowState) -> dict[str, Any]:

    try:
        print("\n[AutoXLab] Data Preparation Agent started")

        result = data_agent.run(state)

        return {
            **result,
            "status": "data_preparation_completed"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"Data preparation failed: {str(e)}"
        }


# ============================================================
# FEATURE ENGINEERING NODE
# ============================================================

def feature_engineering_node(state: WorkflowState) -> dict[str, Any]:

    try:
        print("\n[AutoXLab] Feature Engineering Agent started")

        result = feature_agent.run(state)

        return {
            **result,
            "status": "feature_engineering_completed"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"Feature engineering failed: {str(e)}"
        }


# ============================================================
# ML EXPERIMENT NODE
# ============================================================

def ml_experiment_node(state: WorkflowState) -> dict[str, Any]:

    try:
        print("\n[AutoXLab] ML Experiment Agent started")

        result = ml_agent.run(state)

        current_experiment = state.get("experiment_number", 0)

        return {
            **result,
            "experiment_number": current_experiment + 1,
            "status": "experiment_completed"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"ML experiment failed: {str(e)}"
        }


# ============================================================
# EXPLAINABILITY NODE
# ============================================================

def explainability_node(state: WorkflowState) -> dict[str, Any]:

    try:
        print("\n[AutoXLab] Explainability Agent started")

        result = explainability_agent.run(state)

        return {
            **result,
            "status": "explainability_completed"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"Explainability failed: {str(e)}"
        }

# ============================================================
# DATABASE NODE
# ============================================================
def database_node(
    state: WorkflowState
) -> dict[str, Any]:

    try:

        print(
            "\n[AutoXLab] Database Agent started"
        )

        result = database_agent.run(
            state
        )

        return {
            **result,
            "status":
                "database_logging_completed"
        }

    except Exception as e:

        return {

            "status": "error",

            "error":
                f"Database logging failed: {str(e)}"
        }

# ============================================================
# REPORT GENERATION NODE
# ============================================================

def report_generation_node(state: WorkflowState) -> dict[str, Any]:

    try:
        print("\n[AutoXLab] Report Generation Agent started")

        result = report_agent.run(state)

        return {
            **result,
            "status": "report_completed"
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"Report generation failed: {str(e)}"
        }


# ============================================================
# HUMAN APPROVAL NODE
# ============================================================

def human_approval_node(state: WorkflowState) -> dict[str, Any]:

    print("\n[AutoXLab] Waiting for human approval")

    return {
        "status": "waiting_for_human_approval",
        "approval_message": (
            "The ML experiment and report are complete. "
            "Human approval is required."
        )
    }