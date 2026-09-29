import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat as nbf

from typing import Any

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
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
    - Generate confusion matrix for classification
    - Generate actual-vs-predicted plot for regression
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
            state.get("model_results", [])
        )

        if results.empty:
            return None

        # Remove failed models
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

        # Classification -> accuracy
        # Regression -> R2
        if state.get("problem_type") == "classification":
            metric = "accuracy"
        else:
            metric = "r2"

        if metric not in results.columns:
            return None

        results = results.sort_values(
            metric,
            ascending=False
        )

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
            dpi=180
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

        # Make sure feature names match X_train
        if len(feature_names) != X_train.shape[1]:

            feature_names = [
                f"Feature_{i + 1}"
                for i in range(
                    X_train.shape[1]
                )
            ]

        try:

            importance = permutation_importance(
                model,
                X_train,
                y_train,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            values = importance.importances_mean

            # Top 15 features
            indexes = np.argsort(
                values
            )[-15:]

            names = [
                feature_names[i]
                for i in indexes
            ]

            scores = values[indexes]

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
                dpi=180
            )

            plt.close()

            return path

        except Exception as exc:

            print(
                f"Feature importance failed: {exc}"
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

        if state.get("problem_type") == "classification":

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

            # Add numbers inside matrix
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
                [min_value, max_value],
                [min_value, max_value]
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
            dpi=180
        )

        plt.close()

        return path

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

        metrics_html = ""

        for key, value in state.get(
            "metrics",
            {}
        ).items():

            metrics_html += f"""
            <tr>
                <td>{key}</td>
                <td>{value:.4f}</td>
            </tr>
            """

        evaluation_image = (
            "confusion_matrix.png"
            if state.get("problem_type")
            == "classification"
            else "prediction_plot.png"
        )

        html = f"""
<!DOCTYPE html>

<html>

<head>

    <title>
        AutoXLab Experiment Report
    </title>

    <style>

        body {{
            font-family: Arial;
            margin: 40px;
            background: #f5f7fb;
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

        img {{
            max-width: 850px;
            width: 100%;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
        }}

        td, th {{
            padding: 10px;
            border: 1px solid #ddd;
        }}

    </style>

</head>

<body>

    <div class="card">

        <h1>
            AutoXLab Autonomous
            Data Science Report
        </h1>

        <p>
            Run ID:
            {state.get("run_id", "N/A")}
        </p>

    </div>


    <div class="card">

        <h2>Dataset</h2>

        <p>
            Rows:
            {state.get("dataset_shape", (0, 0))[0]}
        </p>

        <p>
            Columns:
            {state.get("dataset_shape", (0, 0))[1]}
        </p>

        <p>
            Target:
            {state.get("target_column", "N/A")}
        </p>

        <p>
            Problem:
            {state.get("problem_type", "N/A")}
        </p>

    </div>


    <div class="card">

        <h2>Best Model</h2>

        <h3>
            {state.get("best_model_name", "N/A")}
        </h3>

        <table>

            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>

            {metrics_html}

        </table>

    </div>


    <div class="card">

        <h2>Model Comparison</h2>

        <img src="model_comparison.png">

    </div>


    <div class="card">

        <h2>Feature Importance</h2>

        <img src="feature_importance.png">

    </div>


    <div class="card">

        <h2>Evaluation</h2>

        <img src="{evaluation_image}">

    </div>


    <div class="card">

        <h2>Recommendation</h2>

        <p>
            {state.get("recommendation", "")}
        </p>

    </div>

</body>

</html>
"""

        with open(
            html_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(html)

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
            pagesize=letter
        )

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "AutoXLab Autonomous "
                "Data Science Report",
                styles["Title"]
            )
        )

        story.append(
            Spacer(1, 20)
        )

        story.append(
            Paragraph(
                f"Run ID: "
                f"{state.get('run_id', 'N/A')}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Target: "
                f"{state.get('target_column', 'N/A')}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Problem Type: "
                f"{state.get('problem_type', 'N/A')}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Best Model: "
                f"{state.get('best_model_name', 'N/A')}",
                styles["Heading2"]
            )
        )

        # --------------------------------------------------------
        # Metrics table
        # --------------------------------------------------------

        data = [
            ["Metric", "Value"]
        ]

        for key, value in state.get(
            "metrics",
            {}
        ).items():

            data.append(
                [
                    key,
                    f"{value:.4f}"
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
                    )
                ]
            )
        )

        story.append(
            table
        )

        story.append(
            Spacer(1, 20)
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

        notebook = nbf.v4.new_notebook()

        notebook["cells"] = [

            nbf.v4.new_markdown_cell(
                "# AutoXLab Experiment"
            ),

            nbf.v4.new_code_cell(
                f"""
import pandas as pd

df = pd.read_csv(
    r"{state['cleaned_dataset_path']}"
)

df.head()
"""
            ),

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

metrics = {repr(state.get("metrics", {}))}

metrics
"""
            ),

            nbf.v4.new_markdown_cell(
                "## Selected Features"
            ),

            nbf.v4.new_code_cell(
                f"""
selected_features = {
                    repr(
                        state.get(
                            "selected_features",
                            []
                        )
                    )
                }

selected_features
"""
            )
        ]

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

        print("\n" + "=" * 60)
        print(
            "AUTOXLAB - EXPLAINABILITY AGENT"
        )
        print("=" * 60)

        cleaned_dataset_path = state.get(
            "cleaned_dataset_path"
        )

        if not cleaned_dataset_path:
            raise ValueError(
                "cleaned_dataset_path is missing."
            )

        if not state.get("best_model_path"):
            raise ValueError(
                "best_model_path is missing."
            )

        run_dir = os.path.dirname(
            cleaned_dataset_path
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        # --------------------------------------------------------
        # MODEL COMPARISON
        # --------------------------------------------------------

        model_comparison = (
            self.create_model_comparison(
                state,
                run_dir
            )
        )

        # --------------------------------------------------------
        # FEATURE IMPORTANCE
        # --------------------------------------------------------

        feature_importance = (
            self.create_feature_importance(
                state,
                run_dir
            )
        )

        # --------------------------------------------------------
        # EVALUATION
        # --------------------------------------------------------

        evaluation = (
            self.create_evaluation_plot(
                state,
                run_dir
            )
        )

        # --------------------------------------------------------
        # RECOMMENDATION
        # --------------------------------------------------------

        recommendation = (
            f"The experiment evaluated "
            f"{len(state.get('model_results', []))} "
            f"model configurations. "
            f"The selected model was "
            f"{state.get('best_model_name', 'N/A')} "
            f"with a score of "
            f"{state.get('best_score', 0.0):.4f}."
        )

        # --------------------------------------------------------
        # HTML REPORT
        # --------------------------------------------------------

        state_for_reports = {
            **state,
            "recommendation": recommendation
        }

        html_report = (
            self.create_html_report(
                state_for_reports,
                run_dir
            )
        )

        # --------------------------------------------------------
        # PDF REPORT
        # --------------------------------------------------------

        pdf_report = (
            self.create_pdf_report(
                state_for_reports,
                run_dir
            )
        )

        # --------------------------------------------------------
        # NOTEBOOK
        # --------------------------------------------------------

        notebook = (
            self.create_notebook(
                state_for_reports,
                run_dir
            )
        )

        # --------------------------------------------------------
        # VISUALIZATION PATHS
        # --------------------------------------------------------

        visualization_paths = [
            path
            for path in [
                model_comparison,
                feature_importance,
                evaluation
            ]
            if path
        ]

        # --------------------------------------------------------
        # RETURN STATE UPDATE
        # --------------------------------------------------------

        return {

            "visualization_paths":
                visualization_paths,

            "model_comparison_path":
                model_comparison or "",

            "feature_importance_path":
                feature_importance or "",

            "confusion_matrix_path":
                evaluation
                if state.get("problem_type")
                == "classification"
                else "",

            "prediction_plot_path":
                evaluation
                if state.get("problem_type")
                == "regression"
                else "",

            "report_path":
                pdf_report,

            "html_report_path":
                html_report,

            "notebook_path":
                notebook,

            "recommendation":
                recommendation,

            "status":
                "explainability_completed",

            "messages": [

                "Model comparison generated.",

                "Feature importance generated.",

                "Model evaluation generated.",

                "HTML report generated.",

                "PDF report generated.",

                "Jupyter notebook generated."
            ],

            "error": ""
        }