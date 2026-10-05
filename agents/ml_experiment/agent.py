import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from core.llm.router import get_llm

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix
)

from graph.state import WorkflowState


class MLExperimentAgent:
    """
    AutoXLab ML Experiment Agent.

    Responsibilities:
    - Load engineered train/test data
    - Select ML models
    - Train multiple models
    - Evaluate models
    - Generate evaluation visualization
    - Compare model performance
    - Select best model
    - Save best model
    - Return results to LangGraph state
    - Support repeated experiments
    """

    def __init__(self):
        pass

    # ============================================================
    # MODEL SELECTION
    # ============================================================

    def get_models(
        self,
        problem_type: str,
        experiment_number: int
    ) -> dict[str, Any]:

        # --------------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------------

        if problem_type == "classification":

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=2000
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    ),

                "Extra Trees":
                    ExtraTreesClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    ),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        random_state=42
                    ),

                "SVM":
                    SVC(
                        probability=True,
                        random_state=42
                    ),

                "KNN":
                    KNeighborsClassifier(
                        n_neighbors=5
                    )
            }

            # ----------------------------------------------------
            # SECOND EXPERIMENT
            # Try stronger tree models
            # ----------------------------------------------------

            if experiment_number > 1:

                models["Random Forest"] = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=experiment_number,
                    class_weight="balanced"
                )

                models["Extra Trees"] = ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=experiment_number,
                    class_weight="balanced"
                )

            return models

        # --------------------------------------------------------
        # REGRESSION
        # --------------------------------------------------------

        models = {

            "Linear Regression":
                LinearRegression(),

            "Ridge Regression":
                Ridge(
                    alpha=1.0
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "Extra Trees":
                ExtraTreesRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                ),

            "SVR":
                SVR()
        }

        # --------------------------------------------------------
        # SECOND EXPERIMENT
        # --------------------------------------------------------

        if experiment_number > 1:

            models["Random Forest"] = RandomForestRegressor(
                n_estimators=500,
                random_state=experiment_number
            )

        return models

    # ============================================================
    # MAIN AGENT
    # ============================================================

    def run(
        self,
        state: WorkflowState
    ) -> dict[str, Any]:

        print("\n" + "=" * 60)
        print("AUTOXLAB - ML EXPERIMENT AGENT")
        print("=" * 60)

        # --------------------------------------------------------
        # VALIDATE STATE
        # --------------------------------------------------------

        cleaned_dataset_path = state.get(
            "cleaned_dataset_path"
        )

        problem_type = state.get(
            "problem_type"
        )

        if not cleaned_dataset_path:
            raise ValueError(
                "cleaned_dataset_path is missing from WorkflowState."
            )

        if not problem_type:
            raise ValueError(
                "problem_type is missing from WorkflowState."
            )

        # --------------------------------------------------------
        # EXPERIMENT NUMBER
        # --------------------------------------------------------

        experiment_number = state.get(
            "experiment_number",
            1
        )

        print(
            f"Experiment Number: {experiment_number}"
        )

        print(
            f"Problem Type: {problem_type}"
        )

        # --------------------------------------------------------
        # FIND ARTIFACT DIRECTORY
        # --------------------------------------------------------

        run_dir = os.path.dirname(
            cleaned_dataset_path
        )

        # --------------------------------------------------------
        # LOAD TRAINING DATA
        # --------------------------------------------------------

        X_train_path = os.path.join(
            run_dir,
            "X_train.npy"
        )

        X_test_path = os.path.join(
            run_dir,
            "X_test.npy"
        )

        y_train_path = os.path.join(
            run_dir,
            "y_train.npy"
        )

        y_test_path = os.path.join(
            run_dir,
            "y_test.npy"
        )

        if not os.path.exists(X_train_path):
            raise FileNotFoundError(
                f"Training features not found: {X_train_path}"
            )

        if not os.path.exists(X_test_path):
            raise FileNotFoundError(
                f"Testing features not found: {X_test_path}"
            )

        if not os.path.exists(y_train_path):
            raise FileNotFoundError(
                f"Training target not found: {y_train_path}"
            )

        if not os.path.exists(y_test_path):
            raise FileNotFoundError(
                f"Testing target not found: {y_test_path}"
            )

        # --------------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------------

        X_train = np.load(
            X_train_path
        )

        X_test = np.load(
            X_test_path
        )

        y_train = np.load(
            y_train_path,
            allow_pickle=True
        )

        y_test = np.load(
            y_test_path,
            allow_pickle=True
        )

        print(
            f"X_train shape: {X_train.shape}"
        )

        print(
            f"X_test shape: {X_test.shape}"
        )

        print(
            f"y_train shape: {y_train.shape}"
        )

        print(
            f"y_test shape: {y_test.shape}"
        )

        # --------------------------------------------------------
        # GET MODELS
        # --------------------------------------------------------

        models = self.get_models(
            problem_type,
            experiment_number
        )

        print(
            f"\nModels to test: {len(models)}"
        )

        # --------------------------------------------------------
        # EXPERIMENT VARIABLES
        # --------------------------------------------------------

        results = []

        best_model = None

        best_model_name = None

        best_score = -float("inf")

        # --------------------------------------------------------
        # TRAIN EVERY MODEL
        # --------------------------------------------------------

        for name, model in models.items():

            print("\n" + "-" * 50)

            print(
                f"Training: {name}"
            )

            try:

                # --------------------------------------------
                # TRAIN
                # --------------------------------------------

                model.fit(
                    X_train,
                    y_train
                )

                # --------------------------------------------
                # PREDICT
                # --------------------------------------------

                predictions = model.predict(
                    X_test
                )

                # ==================================================
                # CLASSIFICATION
                # ==================================================

                if problem_type == "classification":

                    accuracy = accuracy_score(
                        y_test,
                        predictions
                    )

                    precision = precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )

                    result = {

                        "model": name,

                        "accuracy":
                            float(accuracy),

                        "precision":
                            float(precision),

                        "recall":
                            float(recall),

                        "f1":
                            float(f1)
                    }

                    # F1 is used for model comparison
                    score = f1

                    # ----------------------------------------
                    # ROC-AUC
                    # ----------------------------------------

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        try:

                            probabilities = (
                                model.predict_proba(
                                    X_test
                                )
                            )

                            if len(
                                np.unique(y_test)
                            ) == 2:

                                auc = roc_auc_score(
                                    y_test,
                                    probabilities[:, 1]
                                )

                                result["roc_auc"] = float(
                                    auc
                                )

                        except Exception:
                            pass

                # ==================================================
                # REGRESSION
                # ==================================================

                else:

                    mae = mean_absolute_error(
                        y_test,
                        predictions
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test,
                            predictions
                        )
                    )

                    r2 = r2_score(
                        y_test,
                        predictions
                    )

                    result = {

                        "model": name,

                        "mae":
                            float(mae),

                        "rmse":
                            float(rmse),

                        "r2":
                            float(r2)
                    }

                    # R2 is used for model comparison
                    score = r2

                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                results.append(
                    result
                )

                print(
                    f"Score: {score:.4f}"
                )

                # ------------------------------------------------
                # CHECK BEST MODEL
                # ------------------------------------------------

                if score > best_score:

                    best_score = score

                    best_model = model

                    best_model_name = name

            except Exception as exc:

                print(
                    f"Model failed: {name}"
                )

                print(
                    f"Error: {exc}"
                )

                results.append({

                    "model": name,

                    "error": str(exc)
                })

        # --------------------------------------------------------
        # CHECK WHETHER ANY MODEL WORKED
        # --------------------------------------------------------

        if best_model is None:

            raise RuntimeError(
                "No model could be trained successfully."
            )

        print("\n" + "=" * 60)

        print(
            f"BEST MODEL: {best_model_name}"
        )

        print(
            f"BEST SCORE: {best_score:.4f}"
        )

        # --------------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------------

        best_model_path = os.path.join(
            run_dir,
            "best_model.joblib"
        )

        joblib.dump(
            best_model,
            best_model_path
        )

        print(
            f"Best model saved to: "
            f"{best_model_path}"
        )

        # --------------------------------------------------------
        # BEST MODEL PREDICTIONS
        # --------------------------------------------------------

        predictions = best_model.predict(
            X_test
        )

        # ========================================================
        # GENERATE EVALUATION IMAGE
        # ========================================================

        evaluation_image_path = os.path.join(
            run_dir,
            "evaluation.png"
        )

        print("\n" + "=" * 60)
        print("GENERATING EVALUATION VISUALIZATION")
        print("=" * 60)

        try:

            plt.figure(
                figsize=(8, 6)
            )

            # ====================================================
            # CLASSIFICATION
            # ====================================================

            if problem_type == "classification":

                cm = confusion_matrix(
                    y_test,
                    predictions
                )

                plt.imshow(cm)

                plt.title(
                    f"Confusion Matrix - {best_model_name}"
                )

                plt.xlabel(
                    "Predicted"
                )

                plt.ylabel(
                    "Actual"
                )

                # Add numbers inside confusion matrix
                for i in range(
                    cm.shape[0]
                ):

                    for j in range(
                        cm.shape[1]
                    ):

                        plt.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center"
                        )

                plt.colorbar()

            # ====================================================
            # REGRESSION
            # ====================================================

            else:

                plt.scatter(
                    y_test,
                    predictions,
                    alpha=0.7
                )

                # Perfect prediction line
                minimum = min(
                    np.min(y_test),
                    np.min(predictions)
                )

                maximum = max(
                    np.max(y_test),
                    np.max(predictions)
                )

                plt.plot(
                    [minimum, maximum],
                    [minimum, maximum],
                    linestyle="--"
                )

                plt.xlabel(
                    "Actual Values"
                )

                plt.ylabel(
                    "Predicted Values"
                )

                plt.title(
                    f"Actual vs Predicted - {best_model_name}"
                )

            plt.tight_layout()

            plt.savefig(
                evaluation_image_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Evaluation image saved to: "
                f"{evaluation_image_path}"
            )

        except Exception as evaluation_error:

            print(
                "Evaluation image generation failed."
            )

            print(
                f"Error: {evaluation_error}"
            )

            evaluation_image_path = ""

        # --------------------------------------------------------
        # FINAL METRICS
        # --------------------------------------------------------

        metrics = {}

        if problem_type == "classification":

            metrics = {

                "accuracy":
                    float(
                        accuracy_score(
                            y_test,
                            predictions
                        )
                    ),

                "precision":
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                "recall":
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                "f1":
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    )
            }

            # ROC-AUC for binary classification
            if (
                hasattr(
                    best_model,
                    "predict_proba"
                )
                and len(
                    np.unique(y_test)
                ) == 2
            ):

                try:

                    probabilities = (
                        best_model.predict_proba(
                            X_test
                        )
                    )

                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_test,
                            probabilities[:, 1]
                        )
                    )

                except Exception:
                    pass

        else:

            metrics = {

                "r2":
                    float(
                        r2_score(
                            y_test,
                            predictions
                        )
                    ),

                "mae":
                    float(
                        mean_absolute_error(
                            y_test,
                            predictions
                        )
                    ),

                "rmse":
                    float(
                        np.sqrt(
                            mean_squared_error(
                                y_test,
                                predictions
                            )
                        )
                    )
            }

        # --------------------------------------------------------
        # GENERATE MODEL RECOMMENDATION
        # --------------------------------------------------------

        if best_score >= 0.90:

            recommendation = (
                "The model achieved excellent performance "
                "and may be suitable for further validation."
            )

        elif best_score >= 0.80:

            recommendation = (
                "The model achieved acceptable performance. "
                "Validate it on additional unseen data before deployment."
            )

        else:

            recommendation = (
                "The model performance is below the required threshold. "
                "Further experimentation is recommended."
            )

        # --------------------------------------------------------
        # GENERATE AI ANALYSIS
        # --------------------------------------------------------

        llm_state = {
            **state,
            "best_model_name": best_model_name,
            "best_score": float(best_score),
            "metrics": metrics,
            "recommendation": recommendation,
            "evaluation_image_path": evaluation_image_path
        }

        try:

            llm_analysis = generate_ai_analysis(
                llm_state
            )

            print("=" * 60)
            print("AI ANALYSIS GENERATED SUCCESSFULLY")
            print(llm_analysis)
            print("=" * 60)

        except Exception as e:

            print("=" * 60)
            print("LLM ERROR:")
            print(type(e).__name__)
            print(str(e))
            print("=" * 60)

            llm_analysis = (
                f"LLM analysis failed: "
                f"{type(e).__name__}: {str(e)}"
            )

        # --------------------------------------------------------
        # RETURN LANGGRAPH STATE UPDATE
        # --------------------------------------------------------

        return {

            "model_results": results,

            "best_model_name":
                best_model_name,

            "best_score":
                float(best_score),

            "best_model_path":
                best_model_path,

            # IMPORTANT:
            # Evaluation image path is returned to WorkflowState
            "evaluation_image_path":
                evaluation_image_path,

            # Extra names for compatibility with report generator
            "evaluation_plot_path":
                evaluation_image_path,

            "evaluation_image":
                evaluation_image_path,

            "metrics":
                metrics,

            "recommendation":
                recommendation,

            "llm_analysis":
                llm_analysis,

            "status":
                "ml_experiment_completed",

            "messages": [

                f"Experiment {experiment_number} completed.",

                f"Tested {len(models)} models.",

                f"Best model: {best_model_name}",

                f"Best score: {best_score:.4f}",

                "Best model saved successfully.",

                (
                    f"Evaluation image saved: "
                    f"{evaluation_image_path}"
                )
            ],

            "error": ""
        }


# ===============================================================
# LLM FUNCTION
# ===============================================================

def generate_ai_analysis(
    state
):

    provider = state.get(
        "llm_provider",
        "openai"
    )

    llm = get_llm(
        provider
    )

    prompt = f"""
You are an AI data science assistant.

Analyze the results of this machine learning experiment.

Problem type:
{state.get("problem_type", "unknown")}

Best model:
{state.get("best_model_name", "unknown")}

Best score:
{state.get("best_score", 0.0)}

Metrics:
{state.get("metrics", {})}

Recommendation:
{state.get("recommendation", "")}

Give a short, clear explanation of:

1. Which model performed best
2. How good the score is
3. What the metrics mean
4. Whether the model should be considered for further validation
"""

    response = llm.invoke(
        prompt
    )

    return response.content