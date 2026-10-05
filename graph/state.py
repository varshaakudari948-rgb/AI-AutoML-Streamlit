from typing import TypedDict, Any


class WorkflowState(TypedDict, total=False):

    # ========================================================
    # DATASET
    # ========================================================

    dataset_path: str
    cleaned_dataset_path: str

    target_column: str
    problem_type: str

    dataset_shape: tuple[int, int]

    original_columns: list[str]
    feature_columns: list[str]

    numeric_columns: list[str]
    categorical_columns: list[str]

    missing_values_before: dict[str, int]
    missing_values_after: dict[str, int]

    duplicates_removed: int
    outliers_detected: int

    llm_provider: str
    llm_recommendation: str
    llm_analysis: str

    y_test: list[Any]
    predictions: list[Any]
    prediction_preview: list[dict[str, Any]]


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    preprocessor_path: str

    feature_matrix_path: str
    target_vector_path: str

    X_train_path: str
    X_test_path: str

    y_train_path: str
    y_test_path: str

    selected_features: list[str]
    all_transformed_features: list[str]

    feature_scores: dict[str, float]

    scaling_applied: bool
    encoding_applied: bool

    dimensionality_reduction: bool
    use_pca: bool

    explained_variance: float


    # ========================================================
    # MACHINE LEARNING EXPERIMENT
    # ========================================================

    experiment_number: int

    model_results: list[dict[str, Any]]

    best_model_name: str
    best_score: float

    best_model_path: str

    metrics: dict[str, float]


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    visualization_paths: list[str]

    confusion_matrix_path: str

    feature_importance_path: str

    model_comparison_path: str

    prediction_plot_path: str


    # ========================================================
    # REPORT GENERATION
    # ========================================================

    report_path: str

    html_report_path: str

    notebook_path: str

    recommendation: str


    # ========================================================
    # RETRAINING
    # ========================================================

    retrain_count: int

    max_retrains: int

    minimum_score: float

    should_retrain: bool


    # ========================================================
    # HUMAN APPROVAL
    # ========================================================

    human_approved: bool

    approval_message: str


    # ========================================================
    # DATABASE / MEMORY
    # ========================================================

    run_id: str


    # ========================================================
    # WORKFLOW CONTROL
    # ========================================================

    status: str

    messages: list[str]

    error: str