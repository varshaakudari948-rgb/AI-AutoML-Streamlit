import os
import base64
import html
from pathlib import Path
from typing import Any

from graph.state import WorkflowState


class ReportGenerationAgent:

    def __init__(self):

        # ----------------------------------------------------
        # REPORT OUTPUT DIRECTORY
        # ----------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        self.output_dir = (
            self.project_root
            / "artifacts"
            / "reports"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # RESOLVE IMAGE PATH
    # ========================================================

    def _resolve_image_path(
        self,
        image_path
    ):

        if not image_path:
            return None

        try:

            path = Path(str(image_path))

            # ------------------------------------------------
            # Absolute path
            # ------------------------------------------------

            if path.is_absolute() and path.exists():
                return path

            # ------------------------------------------------
            # Relative to project root
            # ------------------------------------------------

            project_path = (
                self.project_root / path
            )

            if project_path.exists():
                return project_path

            # ------------------------------------------------
            # Relative to current working directory
            # ------------------------------------------------

            cwd_path = (
                Path.cwd() / path
            )

            if cwd_path.exists():
                return cwd_path

        except Exception as e:

            print(
                f"[Report] Path resolution error: {e}"
            )

        return None

    # ========================================================
    # SEARCH IMAGE PATH INSIDE STATE
    # ========================================================

    def _search_state_for_image(
        self,
        state,
        keywords
    ):

        found_paths = []

        def search(value, key_name=""):

            # ------------------------------------------------
            # Dictionary
            # ------------------------------------------------

            if isinstance(value, dict):

                for key, item in value.items():

                    search(
                        item,
                        str(key)
                    )

            # ------------------------------------------------
            # List / Tuple
            # ------------------------------------------------

            elif isinstance(
                value,
                (list, tuple)
            ):

                for item in value:

                    search(
                        item,
                        key_name
                    )

            # ------------------------------------------------
            # String
            # ------------------------------------------------

            elif isinstance(
                value,
                str
            ):

                lower_key = key_name.lower()

                lower_value = value.lower()

                image_extension = (
                    lower_value.endswith(".png")
                    or lower_value.endswith(".jpg")
                    or lower_value.endswith(".jpeg")
                    or lower_value.endswith(".webp")
                )

                keyword_match = any(
                    keyword.lower() in lower_key
                    or keyword.lower() in lower_value
                    for keyword in keywords
                )

                if image_extension and keyword_match:

                    resolved = (
                        self._resolve_image_path(value)
                    )

                    if resolved:

                        found_paths.append(
                            resolved
                        )

        search(state)

        if found_paths:

            return found_paths[0]

        return None

    # ========================================================
    # SEARCH PROJECT FOR IMAGE
    # ========================================================

    def _search_project_for_image(
        self,
        keywords
    ):

        search_directories = [

            self.project_root / "plots",

            self.project_root / "artifacts",

            self.project_root / "data",

        ]

        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        }

        # ----------------------------------------------------
        # Search common plot directories
        # ----------------------------------------------------

        for directory in search_directories:

            if not directory.exists():
                continue

            try:

                for file_path in directory.rglob("*"):

                    if not file_path.is_file():
                        continue

                    if (
                        file_path.suffix.lower()
                        not in image_extensions
                    ):
                        continue

                    filename = (
                        file_path.name.lower()
                    )

                    if all(
                        keyword.lower() in filename
                        for keyword in keywords
                    ):

                        return file_path

            except Exception as e:

                print(
                    f"[Report] Search warning: {e}"
                )

        # ----------------------------------------------------
        # Second search:
        # match ANY keyword
        # ----------------------------------------------------

        for directory in search_directories:

            if not directory.exists():
                continue

            try:

                for file_path in directory.rglob("*"):

                    if not file_path.is_file():
                        continue

                    if (
                        file_path.suffix.lower()
                        not in image_extensions
                    ):
                        continue

                    filename = (
                        file_path.name.lower()
                    )

                    if any(
                        keyword.lower() in filename
                        for keyword in keywords
                    ):

                        return file_path

            except Exception as e:

                print(
                    f"[Report] Search warning: {e}"
                )

        return None

    # ========================================================
    # GET IMAGE
    # ========================================================

    def _get_image(
        self,
        state,
        state_keys,
        keywords
    ):

        # ----------------------------------------------------
        # First: check exact state keys
        # ----------------------------------------------------

        for key in state_keys:

            value = state.get(
                key,
                ""
            )

            if value:

                resolved = (
                    self._resolve_image_path(
                        value
                    )
                )

                if resolved:

                    print(
                        f"[Report] Found {key}: {resolved}"
                    )

                    return resolved

        # ----------------------------------------------------
        # Second: search the complete state
        # ----------------------------------------------------

        resolved = (
            self._search_state_for_image(
                state,
                keywords
            )
        )

        if resolved:

            print(
                f"[Report] Found image in state: {resolved}"
            )

            return resolved

        # ----------------------------------------------------
        # Third: search project directories
        # ----------------------------------------------------

        resolved = (
            self._search_project_for_image(
                keywords
            )
        )

        if resolved:

            print(
                f"[Report] Found image in project: {resolved}"
            )

            return resolved

        print(
            f"[Report] Could not find image for: "
            f"{keywords}"
        )

        return None

    # ========================================================
    # IMAGE TO BASE64
    # ========================================================

    def _image_to_base64(
        self,
        image_path
    ):

        """
        Convert an image file into a Base64 data URL.

        This makes the HTML report self-contained.
        """

        if not image_path:

            return ""

        try:

            image_path = Path(
                image_path
            )

            if not image_path.exists():

                print(
                    f"[Report] Image not found: "
                    f"{image_path}"
                )

                return ""

            with open(
                image_path,
                "rb"
            ) as image_file:

                encoded = (
                    base64.b64encode(
                        image_file.read()
                    )
                    .decode("utf-8")
                )

            extension = (
                image_path.suffix.lower()
            )

            if extension == ".jpg":
                mime_type = "image/jpeg"

            elif extension == ".jpeg":
                mime_type = "image/jpeg"

            elif extension == ".webp":
                mime_type = "image/webp"

            else:
                mime_type = "image/png"

            return (
                f"data:{mime_type};base64,{encoded}"
            )

        except Exception as e:

            print(
                f"[Report] Could not load image "
                f"{image_path}: {e}"
            )

            return ""

    # ========================================================
    # HTML IMAGE SECTION
    # ========================================================

    def _html_image_section(
        self,
        title,
        description,
        image_data
    ):

        if image_data:

            return f"""
            <div class="card">

                <h2>{html.escape(title)}</h2>

                <p>
                    {html.escape(description)}
                </p>

                <div class="image-container">

                    <img
                        src="{image_data}"
                        alt="{html.escape(title)}"
                    >

                </div>

            </div>
            """

        else:

            return f"""
            <div class="card">

                <h2>{html.escape(title)}</h2>

                <p>
                    {html.escape(description)}
                </p>

                <p class="missing">
                    Image could not be found.
                </p>

            </div>
            """

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(
        self,
        state: WorkflowState
    ) -> dict[str, Any]:

        print(
            "\n" + "=" * 60
        )

        print(
            "AUTOXLAB - REPORT GENERATION AGENT"
        )

        print(
            "=" * 60
        )

        # ====================================================
        # GET BASIC STATE VALUES
        # ====================================================

        prediction_preview = state.get(
            "prediction_preview",
            []
        )

        print(
            "PREDICTION PREVIEW:",
            prediction_preview
        )

        run_id = state.get(
            "run_id",
            "unknown_run"
        )

        best_model = state.get(
            "best_model_name",
            "Unknown"
        )

        best_score = state.get(
            "best_score",
            0.0
        )

        problem_type = state.get(
            "problem_type",
            "Unknown"
        )

        metrics = state.get(
            "metrics",
            {}
        )

        recommendation = state.get(
            "recommendation",
            "No recommendation available."
        )

        # ====================================================
        # FIND PLOT IMAGES
        # ====================================================

        print(
            "\n[Report] Searching for generated plots..."
        )

        # ----------------------------------------------------
        # Model Comparison
        # ----------------------------------------------------

        model_comparison_path = self._get_image(
            state,

            [
                "model_comparison_plot",
                "model_comparison_path",
                "model_comparison",
                "comparison_plot"
            ],

            [
                "model_comparison",
                "model-comparison",
                "comparison"
            ]
        )

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        feature_importance_path = self._get_image(
            state,

            [
                "feature_importance_plot",
                "feature_importance_path",
                "feature_importance",
                "importance_plot"
            ],

            [
                "feature_importance",
                "feature-importance",
                "importance"
            ]
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        shap_path = self._get_image(
            state,

            [
                "shap_plot",
                "shap_path",
                "shap_image",
                "shap_explanation"
            ],

            [
                "shap"
            ]
        )

        # ----------------------------------------------------
        # LIME
        # ----------------------------------------------------

        lime_path = self._get_image(
            state,

            [
                "lime_plot",
                "lime_path",
                "lime_image",
                "lime_explanation"
            ],

            [
                "lime"
            ]
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation_path = self._get_image(
            state,

            [
                "evaluation_plot",
                "evaluation_path",
                "evaluation_image",
                "model_evaluation_plot"
            ],

            [
                "evaluation",
                "confusion_matrix",
                "roc_curve"
            ]
        )

        # ====================================================
        # CONVERT IMAGES TO BASE64
        # ====================================================

        model_comparison_img = (
            self._image_to_base64(
                model_comparison_path
            )
        )

        feature_importance_img = (
            self._image_to_base64(
                feature_importance_path
            )
        )

        shap_img = (
            self._image_to_base64(
                shap_path
            )
        )

        lime_img = (
            self._image_to_base64(
                lime_path
            )
        )

        evaluation_img = (
            self._image_to_base64(
                evaluation_path
            )
        )

        # ====================================================
        # HTML REPORT PATH
        # ====================================================

        html_path = (
            self.output_dir
            / f"{run_id}_report.html"
        )

        # ====================================================
        # METRICS TABLE
        # ====================================================

        metrics_html = ""

        if isinstance(
            metrics,
            dict
        ):

            for key, value in metrics.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    metrics_html += f"""
                    <tr>

                        <td>
                            {html.escape(str(key))}
                        </td>

                        <td>
                            {value:.4f}
                        </td>

                    </tr>
                    """

                else:

                    metrics_html += f"""
                    <tr>

                        <td>
                            {html.escape(str(key))}
                        </td>

                        <td>
                            {html.escape(str(value))}
                        </td>

                    </tr>
                    """

        # ====================================================
        # PREDICTION TABLE
        # ====================================================

        prediction_rows_html = ""

        for row in prediction_preview:

            prediction_rows_html += f"""
            <tr>

                <td>
                    {html.escape(
                        str(row.get("sample", ""))
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row.get("actual", ""))
                    )}
                </td>

                <td>
                    {html.escape(
                        str(row.get("predicted", ""))
                    )}
                </td>

            </tr>
            """

        # ====================================================
        # IMAGE HTML SECTIONS
        # ====================================================

        model_comparison_section = (
            self._html_image_section(
                "Model Comparison",
                "Comparison of the ML models evaluated during the experiment.",
                model_comparison_img
            )
        )

        feature_importance_section = (
            self._html_image_section(
                "Feature Importance",
                "Importance of the features used by the selected model.",
                feature_importance_img
            )
        )

        shap_section = (
            self._html_image_section(
                "SHAP Prediction Explanation",
                "SHAP explains how individual features contributed to the selected prediction.",
                shap_img
            )
        )

        lime_section = (
            self._html_image_section(
                "LIME Prediction Explanation",
                "LIME explains which features locally influenced the selected prediction.",
                lime_img
            )
        )

        evaluation_section = (
            self._html_image_section(
                "Evaluation",
                "Evaluation visualization generated for the selected model.",
                evaluation_img
            )
        )

        # ====================================================
        # HTML CONTENT
        # ====================================================

        html_content = f"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>
        AutoXLab Experiment Report
    </title>

    <style>

        body {{

            font-family:
                Arial,
                sans-serif;

            margin:
                40px;

            background:
                #f0f2f5;

            color:
                #222;

        }}

        h1 {{

            color:
                #174ea6;

            margin-bottom:
                30px;

        }}

        h2 {{

            color:
                #1f2937;

        }}

        .card {{

            padding:
                25px;

            margin:
                20px 0;

            border-radius:
                12px;

            background:
                white;

            box-shadow:
                0 2px 8px
                rgba(0,0,0,0.08);

        }}

        table {{

            border-collapse:
                collapse;

            width:
                100%;

        }}

        th,
        td {{

            border:
                1px solid #ddd;

            padding:
                10px;

            text-align:
                left;

        }}

        th {{

            background:
                #eeeeee;

        }}

        .image-container {{

            width:
                100%;

            text-align:
                center;

            margin-top:
                20px;

        }}

        .image-container img {{

            max-width:
                100%;

            height:
                auto;

            border:
                1px solid #ddd;

            border-radius:
                8px;

        }}

        .missing {{

            color:
                #b91c1c;

            font-weight:
                bold;

        }}

        .success {{

            color:
                #15803d;

            font-weight:
                bold;

        }}

    </style>

</head>

<body>

    <h1>
        AutoXLab Experiment Report
    </h1>


    <!-- ================================================= -->
    <!-- EXPERIMENT INFORMATION -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            Experiment Information
        </h2>

        <p>
            <b>Run ID:</b>
            {html.escape(str(run_id))}
        </p>

        <p>
            <b>Problem Type:</b>
            {html.escape(str(problem_type))}
        </p>

        <p>
            <b>Best Model:</b>
            {html.escape(str(best_model))}
        </p>

        <p>
            <b>Best Score:</b>
            {best_score:.4f}
        </p>

    </div>


    <!-- ================================================= -->
    <!-- MODEL METRICS -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            Model Metrics
        </h2>

        <table>

            <tr>

                <th>
                    Metric
                </th>

                <th>
                    Value
                </th>

            </tr>

            {metrics_html}

        </table>

    </div>


    <!-- ================================================= -->
    <!-- MODEL COMPARISON -->
    <!-- ================================================= -->

    {model_comparison_section}


    <!-- ================================================= -->
    <!-- FEATURE IMPORTANCE -->
    <!-- ================================================= -->

    {feature_importance_section}


    <!-- ================================================= -->
    <!-- SHAP -->
    <!-- ================================================= -->

    {shap_section}


    <!-- ================================================= -->
    <!-- LIME -->
    <!-- ================================================= -->

    {lime_section}


    <!-- ================================================= -->
    <!-- EVALUATION -->
    <!-- ================================================= -->

    {evaluation_section}


    <!-- ================================================= -->
    <!-- PREDICTIONS -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            Predictions
        </h2>

        <p>
            The table below shows the first
            10 predictions generated by the
            selected best model.
        </p>

        <table>

            <tr>

                <th>
                    Sample
                </th>

                <th>
                    Actual
                </th>

                <th>
                    Predicted
                </th>

            </tr>

            {prediction_rows_html}

        </table>

    </div>


    <!-- ================================================= -->
    <!-- RECOMMENDATION -->
    <!-- ================================================= -->

    <div class="card">

        <h2>
            Recommendation
        </h2>

        <p>
            {html.escape(
                str(recommendation)
            )}
        </p>

    </div>


</body>

</html>
"""

        # ====================================================
        # SAVE HTML
        # ====================================================

        try:

            with open(
                html_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    html_content
                )

            print(
                f"[Report] HTML report saved: "
                f"{html_path}"
            )

        except Exception as e:

            print(
                f"[Report] HTML generation error: {e}"
            )

            raise

        # ====================================================
        # PDF REPORT
        # ====================================================

        pdf_path = (
            self.output_dir
            / f"{run_id}_report.pdf"
        )

        try:

            from reportlab.lib.pagesizes import A4

            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                Image
            )

            from reportlab.lib import colors

            from reportlab.lib.styles import (
                getSampleStyleSheet
            )

            styles = (
                getSampleStyleSheet()
            )

            document = (
                SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4
                )
            )

            elements = []

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            elements.append(
                Paragraph(
                    "AutoXLab Experiment Report",
                    styles["Title"]
                )
            )

            elements.append(
                Spacer(1, 20)
            )

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            elements.append(
                Paragraph(
                    f"Run ID: {html.escape(str(run_id))}",
                    styles["Normal"]
                )
            )

            elements.append(
                Paragraph(
                    f"Problem Type: "
                    f"{html.escape(str(problem_type))}",
                    styles["Normal"]
                )
            )

            elements.append(
                Paragraph(
                    f"Best Model: "
                    f"{html.escape(str(best_model))}",
                    styles["Normal"]
                )
            )

            elements.append(
                Paragraph(
                    f"Best Score: "
                    f"{best_score:.4f}",
                    styles["Normal"]
                )
            )

            elements.append(
                Spacer(1, 20)
            )

            # ------------------------------------------------
            # MODEL METRICS
            # ------------------------------------------------

            elements.append(
                Paragraph(
                    "Model Metrics",
                    styles["Heading2"]
                )
            )

            table_data = [
                [
                    "Metric",
                    "Value"
                ]
            ]

            if isinstance(
                metrics,
                dict
            ):

                for key, value in metrics.items():

                    if isinstance(
                        value,
                        (int, float)
                    ):

                        table_data.append(
                            [
                                str(key),
                                f"{value:.4f}"
                            ]
                        )

                    else:

                        table_data.append(
                            [
                                str(key),
                                str(value)
                            ]
                        )

            table = Table(
                table_data
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
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        )

                    ]
                )
            )

            elements.append(
                table
            )

            elements.append(
                Spacer(1, 20)
            )

            # ------------------------------------------------
            # ADD IMAGE TO PDF
            # ------------------------------------------------

            def add_pdf_image(
                image_path,
                title
            ):

                if not image_path:

                    return

                try:

                    elements.append(
                        Paragraph(
                            title,
                            styles["Heading2"]
                        )
                    )

                    image = Image(
                        str(image_path)
                    )

                    image._restrictSize(
                        500,
                        350
                    )

                    elements.append(
                        image
                    )

                    elements.append(
                        Spacer(1, 20)
                    )

                except Exception as e:

                    print(
                        f"[Report] PDF image error "
                        f"for {title}: {e}"
                    )

            # ------------------------------------------------
            # ADD ALL PLOTS
            # ------------------------------------------------

            add_pdf_image(
                model_comparison_path,
                "Model Comparison"
            )

            add_pdf_image(
                feature_importance_path,
                "Feature Importance"
            )

            add_pdf_image(
                shap_path,
                "SHAP Prediction Explanation"
            )

            add_pdf_image(
                lime_path,
                "LIME Prediction Explanation"
            )

            add_pdf_image(
                evaluation_path,
                "Evaluation"
            )

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            elements.append(
                Paragraph(
                    "Predictions",
                    styles["Heading2"]
                )
            )

            elements.append(
                Paragraph(
                    "The following table shows the first "
                    "10 predictions generated by the "
                    "selected model.",
                    styles["Normal"]
                )
            )

            prediction_table_data = [

                [
                    "Sample",
                    "Actual",
                    "Predicted"
                ]

            ]

            for row in prediction_preview:

                prediction_table_data.append(
                    [

                        str(
                            row.get(
                                "sample",
                                ""
                            )
                        ),

                        str(
                            row.get(
                                "actual",
                                ""
                            )
                        ),

                        str(
                            row.get(
                                "predicted",
                                ""
                            )
                        )

                    ]
                )

            prediction_table = Table(
                prediction_table_data
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
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            8
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

            elements.append(
                prediction_table
            )

            elements.append(
                Spacer(1, 20)
            )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            elements.append(
                Paragraph(
                    "Recommendation",
                    styles["Heading2"]
                )
            )

            elements.append(
                Paragraph(
                    html.escape(
                        str(recommendation)
                    ),
                    styles["Normal"]
                )
            )

            # ------------------------------------------------
            # BUILD PDF
            # ------------------------------------------------

            document.build(
                elements
            )

            print(
                f"[Report] PDF report saved: "
                f"{pdf_path}"
            )

        except Exception as e:

            print(
                f"[Report] PDF generation warning: {e}"
            )

            pdf_path = ""

        # ====================================================
        # RETURN STATE
        # ====================================================

        return {

            "html_report_path":
                str(html_path),

            "report_path":
                str(pdf_path),

            "notebook_path":
                "",

            "status":
                "report_generated",

            "messages":
                state.get(
                    "messages",
                    []
                )
                + [

                    "HTML report generated",

                    "PDF report generated"

                ]

        }