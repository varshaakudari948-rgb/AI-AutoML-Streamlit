from graph.state import WorkflowState


def check_retraining(state: WorkflowState) -> str:
    """
    Decide whether the ML experiment should be retrained.
    """

    best_score = state.get("best_score", 0.0)

    minimum_score = state.get("minimum_score", 0.80)

    retrain_count = state.get("retrain_count", 0)

    max_retrains = state.get("max_retrains", 2)

    # ------------------------------------------------------------
    # Model reached acceptable performance
    # ------------------------------------------------------------

    if best_score >= minimum_score:

        return "explainability"

    # ------------------------------------------------------------
    # Model is below threshold but retraining is still allowed
    # ------------------------------------------------------------

    if retrain_count < max_retrains:

        return "retrain"

    # ------------------------------------------------------------
    # Maximum retraining attempts reached
    # ------------------------------------------------------------

    return "explainability"