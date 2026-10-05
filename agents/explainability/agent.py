import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat as nbf

from typing import Any

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

import shap
from lime.lime_tabular import LimeTabularExplainer

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from graph.state import WorkflowState


class ExplainabilityAgent:
    """
    AutoXLab Explainability Agent.

    Responsibilities:
    - Compare trained models
    - Generate feature importance
    - Generate SHAP explanation
    - Generate LIME explanation
    - Generate confusion matrix for classification
    - Generate actual-vs-predicted plot for regression
    - Generate prediction table
    - Generate HTML report
    - Generate PDF report
    - Generate Jupyter notebook
    - Return artifacts to LangGraph state
    """

    def __init__(self):
        pass

    # ============================================================
    # MODEL COMPARISON
    # ============================================================

    def create_model_comparison(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str | None:

        results = pd.DataFrame(
            state.get(
                "model_results",
                []
            )
        )

        if results.empty:
            return None

        # --------------------------------------------------------
        # Remove failed models
        # --------------------------------------------------------

        if "error" in results.columns:

            results = results[
                ~results.astype(str)
                .apply(
                    lambda row:
                    row.str.contains(
                        "error",
                        case=False
                    ).any(),
                    axis=1
                )
            ]

        if results.empty:
            return None

        # --------------------------------------------------------
        # Select metric
        # --------------------------------------------------------

        if (
            state.get("problem_type")
            == "classification"
        ):
            metric = "accuracy"

        else:
            metric = "r2"

        if metric not in results.columns:
            return None

        results = results.sort_values(
            metric,
            ascending=False
        )

        # --------------------------------------------------------
        # Create plot
        # --------------------------------------------------------

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            results["model"],
            results[metric]
        )

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.ylabel(
            metric.upper()
        )

        plt.title(
            "Model Performance Comparison"
        )

        plt.tight_layout()

        path = os.path.join(
            run_dir,
            "model_comparison.png"
        )

        plt.savefig(
            path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close()

        return path

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    def create_feature_importance(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str | None:

        model_path = state.get(
            "best_model_path"
        )

        if not model_path:
            return None

        try:

            model = joblib.load(
                model_path
            )

            X_train = np.load(
                os.path.join(
                    run_dir,
                    "X_train.npy"
                )
            )

            y_train = np.load(
                os.path.join(
                    run_dir,
                    "y_train.npy"
                ),
                allow_pickle=True
            )

            feature_names = state.get(
                "all_transformed_features",
                []
            )

            # ----------------------------------------------------
            # Make feature names match number of columns
            # ----------------------------------------------------

            if len(feature_names) != X_train.shape[1]:

                feature_names = [
                    f"Feature_{i + 1}"
                    for i in range(
                        X_train.shape[1]
                    )
                ]

            # ----------------------------------------------------
            # Permutation importance
            # ----------------------------------------------------

            importance = permutation_importance(
                model,
                X_train,
                y_train,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            values = (
                importance.importances_mean
            )

            # ----------------------------------------------------
            # Top 15 features
            # ----------------------------------------------------

            number_of_features = min(
                15,
                len(values)
            )

            indexes = np.argsort(
                values
            )[-number_of_features:]

            names = [
                feature_names[i]
                for i in indexes
            ]

            scores = values[
                indexes
            ]

            # ----------------------------------------------------
            # Plot
            # ----------------------------------------------------

            plt.figure(
                figsize=(10, 7)
            )

            plt.barh(
                names,
                scores
            )

            plt.xlabel(
                "Permutation Importance"
            )

            plt.title(
                "Top Feature Importance"
            )

            plt.tight_layout()

            path = os.path.join(
                run_dir,
                "feature_importance.png"
            )

            plt.savefig(
                path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            return path

        except Exception as exc:

            print(
                f"Feature importance failed: {exc}"
            )

            return None

    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    def create_shap_explanation(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str | None:

        try:

            model = joblib.load(
                state["best_model_path"]
            )

            X_train = np.load(
                os.path.join(
                    run_dir,
                    "X_train.npy"
                )
            )

            X_test = np.load(
                os.path.join(
                    run_dir,
                    "X_test.npy"
                )
            )

            feature_names = state.get(
                "all_transformed_features",
                []
            )

            # ----------------------------------------------------
            # Make feature names match data
            # ----------------------------------------------------

            if len(feature_names) != X_train.shape[1]:

                feature_names = [
                    f"Feature_{i + 1}"
                    for i in range(
                        X_train.shape[1]
                    )
                ]

            # ----------------------------------------------------
            # Use small background dataset
            # ----------------------------------------------------

            background_size = min(
                100,
                len(X_train)
            )

            background = X_train[
                :background_size
            ]

            # ----------------------------------------------------
            # Use model.predict to keep SHAP output simple
            # and compatible with classification/regression
            # ----------------------------------------------------

            explainer = shap.Explainer(
                model.predict,
                background,
                feature_names=feature_names
            )

            # ----------------------------------------------------
            # Explain first test sample
            # ----------------------------------------------------

            sample = X_test[
                0:1
            ]

            shap_values = explainer(
                sample
            )

            # ----------------------------------------------------
            # Create waterfall plot
            # ----------------------------------------------------

            plt.figure(
                figsize=(10, 7)
            )

            shap.plots.waterfall(
                shap_values[0],
                max_display=15,
                show=False
            )

            plt.tight_layout()

            path = os.path.join(
                run_dir,
                "shap_prediction.png"
            )

            plt.savefig(
                path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            return path

        except Exception as exc:

            print(
                f"SHAP explanation failed: {exc}"
            )

            return None

    # ============================================================
    # LIME CLASSIFICATION PREDICTION FUNCTION
    # ============================================================

    def _get_lime_prediction_function(
        self,
        model,
        problem_type: str
    ):

        # --------------------------------------------------------
        # Regression
        # --------------------------------------------------------

        if problem_type != "classification":

            return model.predict

        # --------------------------------------------------------
        # Classification with predict_proba
        # --------------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            return model.predict_proba

        # --------------------------------------------------------
        # Classification without predict_proba
        # Use decision_function and convert scores
        # into probabilities.
        # --------------------------------------------------------

        if hasattr(
            model,
            "decision_function"
        ):

            def decision_to_probability(
                X
            ):

                scores = np.asarray(
                    model.decision_function(X)
                )

                # Binary classification
                if scores.ndim == 1:

                    scores = np.clip(
                        scores,
                        -500,
                        500
                    )

                    positive_probability = (
                        1.0
                        /
                        (
                            1.0
                            +
                            np.exp(-scores)
                        )
                    )

                    return np.column_stack(
                        [
                            1.0
                            - positive_probability,

                            positive_probability
                        ]
                    )

                # Multiclass classification
                scores = scores - np.max(
                    scores,
                    axis=1,
                    keepdims=True
                )

                exp_scores = np.exp(
                    scores
                )

                return (
                    exp_scores
                    /
                    np.sum(
                        exp_scores,
                        axis=1,
                        keepdims=True
                    )
                )

            return decision_to_probability

        # --------------------------------------------------------
        # Final fallback
        # --------------------------------------------------------

        raise AttributeError(
            "The selected classification model "
            "does not provide predict_proba "
            "or decision_function."
        )

    # ============================================================
    # LIME EXPLANATION
    # ============================================================

    def create_lime_explanation(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str | None:

        try:

            model = joblib.load(
                state["best_model_path"]
            )

            X_train = np.load(
                os.path.join(
                    run_dir,
                    "X_train.npy"
                )
            )

            X_test = np.load(
                os.path.join(
                    run_dir,
                    "X_test.npy"
                )
            )

            feature_names = state.get(
                "all_transformed_features",
                []
            )

            # ----------------------------------------------------
            # Make feature names match X_train
            # ----------------------------------------------------

            if len(feature_names) != X_train.shape[1]:

                feature_names = [
                    f"Feature_{i + 1}"
                    for i in range(
                        X_train.shape[1]
                    )
                ]

            problem_type = state.get(
                "problem_type",
                "classification"
            )

            # ----------------------------------------------------
            # Create LIME explainer
            # ----------------------------------------------------

            if problem_type == "classification":

                explainer = (
                    LimeTabularExplainer(
                        X_train,
                        feature_names=feature_names,
                        mode="classification",
                        discretize_continuous=True,
                        random_state=42
                    )
                )

                prediction_function = (
                    self._get_lime_prediction_function(
                        model,
                        problem_type
                    )
                )

            else:

                explainer = (
                    LimeTabularExplainer(
                        X_train,
                        feature_names=feature_names,
                        mode="regression",
                        discretize_continuous=True,
                        random_state=42
                    )
                )

                prediction_function = (
                    model.predict
                )

            # ----------------------------------------------------
            # Explain first test sample
            # ----------------------------------------------------

            explanation = (
                explainer.explain_instance(
                    X_test[0],
                    prediction_function,
                    num_features=10
                )
            )

            # ----------------------------------------------------
            # Save LIME HTML
            # ----------------------------------------------------

            html_path = os.path.join(
                run_dir,
                "lime_prediction.html"
            )

            explanation.save_to_file(
                html_path
            )

            # ----------------------------------------------------
            # Save LIME PNG
            # ----------------------------------------------------

            png_path = os.path.join(
                run_dir,
                "lime_prediction.png"
            )

            figure = (
                explanation.as_pyplot_figure()
            )

            figure.set_size_inches(
                10,
                7
            )

            figure.tight_layout()

            figure.savefig(
                png_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(
                figure
            )

            return png_path

        except Exception as exc:

            print(
                f"LIME explanation failed: {exc}"
            )

            return None

    # ============================================================
    # EVALUATION PLOT
    # ============================================================

    def create_evaluation_plot(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str:

        model = joblib.load(
            state["best_model_path"]
        )

        X_test = np.load(
            os.path.join(
                run_dir,
                "X_test.npy"
            )
        )

        y_test = np.load(
            os.path.join(
                run_dir,
                "y_test.npy"
            ),
            allow_pickle=True
        )

        predictions = model.predict(
            X_test
        )

        # ========================================================
        # CLASSIFICATION
        # ========================================================

        if (
            state.get("problem_type")
            == "classification"
        ):

            cm = confusion_matrix(
                y_test,
                predictions
            )

            plt.figure(
                figsize=(7, 6)
            )

            plt.imshow(cm)

            plt.title(
                "Confusion Matrix"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "Actual"
            )

            # ----------------------------------------------------
            # Add values inside matrix
            # ----------------------------------------------------

            for i in range(
                cm.shape[0]
            ):

                for j in range(
                    cm.shape[1]
                ):

                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center"
                    )

            plt.colorbar()

            path = os.path.join(
                run_dir,
                "confusion_matrix.png"
            )

        # ========================================================
        # REGRESSION
        # ========================================================

        else:

            plt.figure(
                figsize=(8, 6)
            )

            plt.scatter(
                y_test,
                predictions,
                alpha=0.7
            )

            min_value = min(
                np.min(y_test),
                np.min(predictions)
            )

            max_value = max(
                np.max(y_test),
                np.max(predictions)
            )

            plt.plot(
                [
                    min_value,
                    max_value
                ],
                [
                    min_value,
                    max_value
                ]
            )

            plt.xlabel(
                "Actual"
            )

            plt.ylabel(
                "Predicted"
            )

            plt.title(
                "Actual vs Predicted"
            )

            path = os.path.join(
                run_dir,
                "prediction_plot.png"
            )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close()

        return path

    # ============================================================
    # GET PREDICTIONS
    # ============================================================

    def get_predictions(
        self,
        state: WorkflowState,
        run_dir: str
    ):

        model = joblib.load(
            state["best_model_path"]
        )

        X_test = np.load(
            os.path.join(
                run_dir,
                "X_test.npy"
            )
        )

        y_test = np.load(
            os.path.join(
                run_dir,
                "y_test.npy"
            ),
            allow_pickle=True
        )

        predictions = model.predict(
            X_test
        )

        return (
            y_test,
            predictions
        )

    # ============================================================
    # HTML REPORT
    # ============================================================

    def create_html_report(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str:

        html_path = os.path.join(
            run_dir,
            "experiment_report.html"
        )

        # --------------------------------------------------------
        # Metrics HTML
        # --------------------------------------------------------

        metrics_html = ""

        for key, value in state.get(
            "metrics",
            {}
        ).items():

            try:
                formatted_value = (
                    f"{float(value):.4f}"
                )

            except (
                ValueError,
                TypeError
            ):

                formatted_value = str(
                    value
                )

            metrics_html += f"""
            <tr>
                <td>{key}</td>
                <td>{formatted_value}</td>
            </tr>
            """

        # --------------------------------------------------------
        # Evaluation image
        # --------------------------------------------------------

        if (
            state.get("problem_type")
            == "classification"
        ):

            evaluation_image = (
                "confusion_matrix.png"
            )

        else:

            evaluation_image = (
                "prediction_plot.png"
            )

        # --------------------------------------------------------
        # Prediction table
        # --------------------------------------------------------

        prediction_rows = ""

        y_test = state.get(
            "y_test",
            []
        )

        predictions = state.get(
            "predictions",
            []
        )

        number_of_predictions = min(
            10,
            len(predictions)
        )

        for i in range(
            number_of_predictions
        ):

            prediction_rows += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{y_test[i]}</td>
                <td>{predictions[i]}</td>
            </tr>
            """

        # --------------------------------------------------------
        # Check generated image files
        # --------------------------------------------------------

        model_comparison_html = ""

        model_comparison_path = os.path.join(
            run_dir,
            "model_comparison.png"
        )

        if os.path.exists(
            model_comparison_path
        ):

            model_comparison_html = """
            <div class="card">

                <h2>Model Comparison</h2>

                <img
                    src="model_comparison.png"
                    alt="Model Comparison"
                >

            </div>
            """

        feature_importance_html = ""

        feature_importance_path = os.path.join(
            run_dir,
            "feature_importance.png"
        )

        if os.path.exists(
            feature_importance_path
        ):

            feature_importance_html = """
            <div class="card">

                <h2>Feature Importance</h2>

                <img
                    src="feature_importance.png"
                    alt="Feature Importance"
                >

            </div>
            """

        shap_html = ""

        shap_path = os.path.join(
            run_dir,
            "shap_prediction.png"
        )

        if os.path.exists(
            shap_path
        ):

            shap_html = """
            <div class="card">

                <h2>
                    SHAP Prediction Explanation
                </h2>

                <p>
                    SHAP explains how individual
                    features contributed to the
                    selected prediction.
                </p>

                <img
                    src="shap_prediction.png"
                    alt="SHAP Explanation"
                >

            </div>
            """

        lime_html = ""

        lime_png_path = os.path.join(
            run_dir,
            "lime_prediction.png"
        )

        if os.path.exists(
            lime_png_path
        ):

            lime_html = """
            <div class="card">

                <h2>
                    LIME Prediction Explanation
                </h2>

                <p>
                    LIME explains which features
                    locally influenced the
                    selected prediction.
                </p>

                <img
                    src="lime_prediction.png"
                    alt="LIME Explanation"
                >

                <p>
                    <a
                        href="lime_prediction.html"
                        target="_blank"
                    >
                        Open Complete LIME Explanation
                    </a>
                </p>

            </div>
            """

        evaluation_html = ""

        evaluation_path = os.path.join(
            run_dir,
            evaluation_image
        )

        if os.path.exists(
            evaluation_path
        ):

            evaluation_html = f"""
            <div class="card">

                <h2>Evaluation</h2>

                <img
                    src="{evaluation_image}"
                    alt="Model Evaluation"
                >

            </div>
            """

        # --------------------------------------------------------
        # Complete HTML
        # --------------------------------------------------------

        html = f"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>
        AutoXLab Experiment Report
    </title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f5f7fb;
            color: #222;
        }}

        .card {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow:
                0 2px 10px
                rgba(0,0,0,.08);
        }}

        h1 {{
            color: #174ea6;
        }}

        h2 {{
            color: #333;
        }}

        img {{
            max-width: 850px;
            width: 100%;
            height: auto;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
        }}

        td,
        th {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }}

        th {{
            background: #f0f0f0;
        }}

        a {{
            color: #174ea6;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

    </style>

</head>

<body>

    <!-- ===================================================== -->
    <!-- HEADER -->
    <!-- ===================================================== -->

    <div class="card">

        <h1>
            AutoXLab Autonomous
            Data Science Report
        </h1>

        <p>
            <strong>Run ID:</strong>
            {state.get("run_id", "N/A")}
        </p>

    </div>

    <!-- ===================================================== -->
    <!-- DATASET -->
    <!-- ===================================================== -->

    <div class="card">

        <h2>Dataset</h2>

        <p>
            <strong>Rows:</strong>
            {state.get("dataset_shape", (0, 0))[0]}
        </p>

        <p>
            <strong>Columns:</strong>
            {state.get("dataset_shape", (0, 0))[1]}
        </p>

        <p>
            <strong>Target:</strong>
            {state.get("target_column", "N/A")}
        </p>

        <p>
            <strong>Problem:</strong>
            {state.get("problem_type", "N/A")}
        </p>

    </div>

    <!-- ===================================================== -->
    <!-- BEST MODEL -->
    <!-- ===================================================== -->

    <div class="card">

        <h2>Best Model</h2>

        <h3>
            {state.get(
                "best_model_name",
                "N/A"
            )}
        </h3>

        <table>

            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>

            {metrics_html}

        </table>

    </div>

    <!-- ===================================================== -->
    <!-- MODEL COMPARISON -->
    <!-- ===================================================== -->

    {model_comparison_html}

    <!-- ===================================================== -->
    <!-- FEATURE IMPORTANCE -->
    <!-- ===================================================== -->

    {feature_importance_html}

    <!-- ===================================================== -->
    <!-- PREDICTIONS -->
    <!-- ===================================================== -->

    <div class="card">

        <h2>Predictions</h2>

        <p>
            The table below shows the first
            10 predictions generated by the
            selected best model.
        </p>

        <table>

            <tr>
                <th>Sample</th>
                <th>Actual</th>
                <th>Predicted</th>
            </tr>

            {prediction_rows}

        </table>

    </div>

    <!-- ===================================================== -->
    <!-- SHAP -->
    <!-- ===================================================== -->

    {shap_html}

    <!-- ===================================================== -->
    <!-- LIME -->
    <!-- ===================================================== -->

    {lime_html}

    <!-- ===================================================== -->
    <!-- EVALUATION -->
    <!-- ===================================================== -->

    {evaluation_html}

    <!-- ===================================================== -->
    <!-- RECOMMENDATION -->
    <!-- ===================================================== -->

    <div class="card">

        <h2>Recommendation</h2>

        <p>
            {state.get(
                "recommendation",
                ""
            )}
        </p>

    </div>

</body>

</html>
"""

        # --------------------------------------------------------
        # Save HTML file
        # --------------------------------------------------------

        with open(
            html_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                html
            )

        return html_path

    # ============================================================
    # PDF REPORT
    # ============================================================

    def create_pdf_report(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str:

        pdf_path = os.path.join(
            run_dir,
            "experiment_report.pdf"
        )

        document = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = (
            getSampleStyleSheet()
        )

        story = []

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------

        story.append(
            Paragraph(
                "AutoXLab Autonomous "
                "Data Science Report",
                styles["Title"]
            )
        )

        story.append(
            Spacer(
                1,
                20
            )
        )

        # --------------------------------------------------------
        # Run ID
        # --------------------------------------------------------

        story.append(
            Paragraph(
                f"Run ID: "
                f"{state.get('run_id', 'N/A')}",
                styles["Normal"]
            )
        )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        story.append(
            Paragraph(
                f"Target: "
                f"{state.get('target_column', 'N/A')}",
                styles["Normal"]
            )
        )

        # --------------------------------------------------------
        # Problem Type
        # --------------------------------------------------------

        story.append(
            Paragraph(
                f"Problem Type: "
                f"{state.get('problem_type', 'N/A')}",
                styles["Normal"]
            )
        )

        # --------------------------------------------------------
        # Best Model
        # --------------------------------------------------------

        story.append(
            Paragraph(
                f"Best Model: "
                f"{state.get('best_model_name', 'N/A')}",
                styles["Heading2"]
            )
        )

        story.append(
            Spacer(
                1,
                15
            )
        )

        # ========================================================
        # METRICS TABLE
        # ========================================================

        data = [
            [
                "Metric",
                "Value"
            ]
        ]

        for key, value in state.get(
            "metrics",
            {}
        ).items():

            try:

                formatted_value = (
                    f"{float(value):.4f}"
                )

            except (
                ValueError,
                TypeError
            ):

                formatted_value = str(
                    value
                )

            data.append(
                [
                    key,
                    formatted_value
                ]
            )

        table = Table(
            data
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.grey
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    )
                ]
            )
        )

        story.append(
            table
        )

        story.append(
            Spacer(
                1,
                20
            )
        )

        # ========================================================
        # RECOMMENDATION
        # ========================================================

        story.append(
            Paragraph(
                "Recommendation",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                state.get(
                    "recommendation",
                    ""
                ),
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(
                1,
                20
            )
        )

        # ========================================================
        # PREDICTIONS
        # ========================================================

        story.append(
            Paragraph(
                "Predictions",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                "The table below shows the first "
                "10 predictions generated by "
                "the selected best model.",
                styles["BodyText"]
            )
        )

        prediction_data = [
            [
                "Sample",
                "Actual",
                "Predicted"
            ]
        ]

        y_test = state.get(
            "y_test",
            []
        )

        predictions = state.get(
            "predictions",
            []
        )

        number_of_predictions = min(
            10,
            len(predictions)
        )

        for i in range(
            number_of_predictions
        ):

            prediction_data.append(
                [
                    str(i + 1),
                    str(y_test[i]),
                    str(predictions[i])
                ]
            )

        prediction_table = Table(
            prediction_data
        )

        prediction_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        1,
                        colors.grey
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    )
                ]
            )
        )

        story.append(
            prediction_table
        )

        story.append(
            Spacer(
                1,
                20
            )
        )

        # ========================================================
        # MODEL COMPARISON
        # ========================================================

        model_comparison_path = os.path.join(
            run_dir,
            "model_comparison.png"
        )

        if os.path.exists(
            model_comparison_path
        ):

            story.append(
                Paragraph(
                    "Model Comparison",
                    styles["Heading2"]
                )
            )

            story.append(
                Image(
                    model_comparison_path,
                    width=440,
                    height=265
                )
            )

            story.append(
                Spacer(
                    1,
                    20
                )
            )

        # ========================================================
        # FEATURE IMPORTANCE
        # ========================================================

        feature_importance_path = os.path.join(
            run_dir,
            "feature_importance.png"
        )

        if os.path.exists(
            feature_importance_path
        ):

            story.append(
                Paragraph(
                    "Feature Importance",
                    styles["Heading2"]
                )
            )

            story.append(
                Image(
                    feature_importance_path,
                    width=440,
                    height=300
                )
            )

            story.append(
                Spacer(
                    1,
                    20
                )
            )

        # ========================================================
        # SHAP
        # ========================================================

        shap_path = os.path.join(
            run_dir,
            "shap_prediction.png"
        )

        if os.path.exists(
            shap_path
        ):

            story.append(
                Paragraph(
                    "SHAP Prediction Explanation",
                    styles["Heading2"]
                )
            )

            story.append(
                Paragraph(
                    "SHAP shows how individual "
                    "features contributed to "
                    "the selected prediction.",
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(
                    1,
                    10
                )
            )

            story.append(
                Image(
                    shap_path,
                    width=440,
                    height=300
                )
            )

            story.append(
                Spacer(
                    1,
                    20
                )
            )

        # ========================================================
        # LIME
        # ========================================================

        lime_path = os.path.join(
            run_dir,
            "lime_prediction.png"
        )

        if os.path.exists(
            lime_path
        ):

            story.append(
                Paragraph(
                    "LIME Prediction Explanation",
                    styles["Heading2"]
                )
            )

            story.append(
                Paragraph(
                    "LIME shows the features "
                    "that locally influenced "
                    "the selected prediction.",
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(
                    1,
                    10
                )
            )

            story.append(
                Image(
                    lime_path,
                    width=440,
                    height=300
                )
            )

            story.append(
                Spacer(
                    1,
                    20
                )
            )

        # ========================================================
        # EVALUATION
        # ========================================================

        if (
            state.get("problem_type")
            == "classification"
        ):

            evaluation_path = os.path.join(
                run_dir,
                "confusion_matrix.png"
            )

            evaluation_title = (
                "Confusion Matrix"
            )

        else:

            evaluation_path = os.path.join(
                run_dir,
                "prediction_plot.png"
            )

            evaluation_title = (
                "Actual vs Predicted"
            )

        if os.path.exists(
            evaluation_path
        ):

            story.append(
                Paragraph(
                    evaluation_title,
                    styles["Heading2"]
                )
            )

            story.append(
                Image(
                    evaluation_path,
                    width=440,
                    height=300
                )
            )

            story.append(
                Spacer(
                    1,
                    20
                )
            )

        # ========================================================
        # BUILD PDF
        # ========================================================

        document.build(
            story
        )

        return pdf_path

    # ============================================================
    # JUPYTER NOTEBOOK
    # ============================================================

    def create_notebook(
        self,
        state: WorkflowState,
        run_dir: str
    ) -> str:

        notebook_path = os.path.join(
            run_dir,
            "autoxlab_experiment.ipynb"
        )

        notebook = (
            nbf.v4.new_notebook()
        )

        # --------------------------------------------------------
        # Prediction values for notebook
        # --------------------------------------------------------

        y_test = state.get(
            "y_test",
            []
        )

        predictions = state.get(
            "predictions",
            []
        )

        notebook["cells"] = [

            # ----------------------------------------------------
            # Title
            # ----------------------------------------------------

            nbf.v4.new_markdown_cell(
                "# AutoXLab Experiment"
            ),

            # ----------------------------------------------------
            # Load dataset
            # ----------------------------------------------------

            nbf.v4.new_code_cell(
                f"""
import pandas as pd

df = pd.read_csv(
    r"{state['cleaned_dataset_path']}"
)

df.head()
"""
            ),

            # ----------------------------------------------------
            # Experiment information
            # ----------------------------------------------------

            nbf.v4.new_code_cell(
                f"""
print(
    "Target:",
    "{state.get('target_column', 'N/A')}"
)

print(
    "Problem:",
    "{state.get('problem_type', 'N/A')}"
)

print(
    "Best model:",
    "{state.get('best_model_name', 'N/A')}"
)

print("Metrics:")

metrics = {repr(
    state.get(
        "metrics",
        {}
    )
)}

metrics
"""
            ),

            # ----------------------------------------------------
            # Selected features
            # ----------------------------------------------------

            nbf.v4.new_markdown_cell(
                "## Selected Features"
            ),

            nbf.v4.new_code_cell(
                f"""
selected_features = {repr(
    state.get(
        "selected_features",
        []
    )
)}

selected_features
"""
            ),

            # ----------------------------------------------------
            # Predictions
            # ----------------------------------------------------

            nbf.v4.new_markdown_cell(
                "## Predictions"
            ),

            nbf.v4.new_code_cell(
                f"""
actual_values = {repr(
    np.asarray(
        y_test
    ).tolist()
)}

predicted_values = {repr(
    np.asarray(
        predictions
    ).tolist()
)}

prediction_table = pd.DataFrame(
    {{
        "Actual": actual_values[:10],
        "Predicted": predicted_values[:10]
    }}
)

prediction_table
"""
            )
        ]

        # --------------------------------------------------------
        # Save notebook
        # --------------------------------------------------------

        with open(
            notebook_path,
            "w",
            encoding="utf-8"
        ) as file:

            nbf.write(
                notebook,
                file
            )

        return notebook_path

    # ============================================================
    # MAIN AGENT
    # ============================================================

    def run(
        self,
        state: WorkflowState
    ) -> dict[str, Any]:

        print(
            "\n" + "=" * 60
        )

        print(
            "AUTOXLAB - EXPLAINABILITY AGENT"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Dataset path
        # --------------------------------------------------------

        cleaned_dataset_path = state.get(
            "cleaned_dataset_path"
        )

        if not cleaned_dataset_path:

            raise ValueError(
                "cleaned_dataset_path is missing."
            )

        # --------------------------------------------------------
        # Best model path
        # --------------------------------------------------------

        if not state.get(
            "best_model_path"
        ):

            raise ValueError(
                "best_model_path is missing."
            )

        # --------------------------------------------------------
        # Run directory
        # --------------------------------------------------------

        run_dir = os.path.dirname(
            cleaned_dataset_path
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        # ========================================================
        # MODEL COMPARISON
        # ========================================================

        model_comparison = (
            self.create_model_comparison(
                state,
                run_dir
            )
        )

        # ========================================================
        # FEATURE IMPORTANCE
        # ========================================================

        feature_importance = (
            self.create_feature_importance(
                state,
                run_dir
            )
        )

        # ========================================================
        # SHAP
        # ========================================================

        shap_explanation = (
            self.create_shap_explanation(
                state,
                run_dir
            )
        )

        # ========================================================
        # LIME
        # ========================================================

        lime_explanation = (
            self.create_lime_explanation(
                state,
                run_dir
            )
        )

        # ========================================================
        # EVALUATION
        # ========================================================

        evaluation = (
            self.create_evaluation_plot(
                state,
                run_dir
            )
        )

        # ========================================================
        # PREDICTIONS
        # ========================================================

        y_test, predictions = (
            self.get_predictions(
                state,
                run_dir
            )
        )
        # --------------------------------------------------------
        # PREDICTION PREVIEW
        # --------------------------------------------------------

        prediction_preview = []

        for i in range(
            min(10, len(predictions))
        ):

            prediction_preview.append(
                {
                    "sample": i + 1,
                    "actual": float(y_test[i]),
                    "predicted": float(predictions[i])
                }
            )
        # ========================================================
        # RECOMMENDATION
        # ========================================================

        recommendation = (
            f"The experiment evaluated "
            f"{len(state.get('model_results', []))} "
            f"model configurations. "
            f"The selected model was "
            f"{state.get('best_model_name', 'N/A')} "
            f"with a score of "
            f"{state.get('best_score', 0.0):.4f}."
        )

        # ========================================================
        # STATE FOR REPORTS
        # ========================================================

        state_for_reports = {
            **state,

            "recommendation":
                recommendation,

            "y_test":
                y_test,

            "predictions":
                predictions
        }

        # ========================================================
        # HTML REPORT
        # ========================================================

        html_report = (
            self.create_html_report(
                state_for_reports,
                run_dir
            )
        )

        # ========================================================
        # PDF REPORT
        # ========================================================

        pdf_report = (
            self.create_pdf_report(
                state_for_reports,
                run_dir
            )
        )

        # ========================================================
        # NOTEBOOK
        # ========================================================

        notebook = (
            self.create_notebook(
                state_for_reports,
                run_dir
            )
        )

        # ========================================================
        # VISUALIZATION PATHS
        # ========================================================

        visualization_paths = [

            path

            for path in [

                model_comparison,

                feature_importance,

                shap_explanation,

                lime_explanation,

                evaluation
            ]

            if path
        ]

        # ========================================================
        # RETURN STATE UPDATE
        # ========================================================

        return {

            "visualization_paths":
                visualization_paths,

            "model_comparison_path":
                model_comparison or "",

            "feature_importance_path":
                feature_importance or "",

            "shap_explanation_path":
                shap_explanation or "",

            "lime_explanation_path":
                lime_explanation or "",

            "confusion_matrix_path":

                evaluation

                if (
                    state.get(
                        "problem_type"
                    )
                    == "classification"
                )

                else "",

            "prediction_plot_path":

                evaluation

                if (
                    state.get(
                        "problem_type"
                    )
                    == "regression"
                )

                else "",

            "report_path":
                pdf_report,

            "html_report_path":
                html_report,

            "notebook_path":
                notebook,

            "recommendation":
                recommendation,
            "y_test":
                np.asarray(
                    y_test
                ).tolist(),

            "predictions":
                np.asarray(
                    predictions
                ).tolist(),

            "prediction_preview":
                prediction_preview,

            "status":
                "explainability_completed",

            "messages": [

                "Model comparison generated.",

                "Feature importance generated.",

                "SHAP explanation generated.",

                "LIME explanation generated.",

                "Predictions generated.",

                "Model evaluation generated.",

                "HTML report generated.",

                "PDF report generated.",

                "Jupyter notebook generated."
            ],

            "error":
                ""
        }