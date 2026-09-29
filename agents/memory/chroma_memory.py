import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CHROMA CONFIGURATION
# ============================================================

USE_CHROMA = (
    os.getenv(
        "USE_CHROMA",
        "true"
    ).lower()
    == "true"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "data/chroma"
)


# ============================================================
# GET CHROMA COLLECTION
# ============================================================

def get_collection():

    if not USE_CHROMA:
        return None

    import chromadb

    os.makedirs(
        CHROMA_PATH,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="autoxlab_experiments"
    )

    return collection


# ============================================================
# SAVE EXPERIMENT MEMORY
# ============================================================

def save_experiment_memory(state):

    collection = get_collection()

    if collection is None:
        return

    document = f"""
    Dataset target: {state['target_column']}

    Problem type: {state['problem_type']}

    Best model: {state['best_model_name']}

    Score: {state['best_score']}

    Metrics: {state['metrics']}

    Selected features:
    {state['selected_features']}

    Recommendation:
    {state.get('recommendation', '')}
    """

    collection.add(

        ids=[
            state["run_id"]
        ],

        documents=[
            document
        ],

        metadatas=[
            {
                "run_id":
                    state["run_id"],

                "best_model":
                    state["best_model_name"],

                "problem_type":
                    state["problem_type"]
            }
        ]
    )


# ============================================================
# SEARCH PREVIOUS EXPERIMENTS
# ============================================================

def search_previous_experiments(
    query: str,
    n_results: int = 5
):

    collection = get_collection()

    if collection is None:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return result.get(
        "documents",
        [[]]
    )[0]