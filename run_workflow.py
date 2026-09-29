import uuid

from graph.workflow import graph


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = {

    "dataset_path":
        "data/raw/dataset.csv",

    "target_column":
        "target",

    "use_pca":
        False,

    "experiment_number":
        1,

    "retrain_count":
        0,

    "max_retrains":
        2,

    "minimum_score":
        0.80,

    "run_id":
        str(uuid.uuid4())
}


# ============================================================
# LANGGRAPH CONFIG
# ============================================================

config = {

    "configurable": {

        "thread_id":
            initial_state["run_id"]
    }
}


# ============================================================
# RUN WORKFLOW
# ============================================================

print("\nStarting AutoXLab workflow...")

result = graph.invoke(
    initial_state,
    config=config
)

print("\n" + "=" * 60)
print("FINAL WORKFLOW STATE")
print("=" * 60)

print(result)