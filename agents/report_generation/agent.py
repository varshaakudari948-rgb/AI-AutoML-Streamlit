import os
from typing import Any

from graph.state import WorkflowState


class ReportGenerationAgent:

    def __init__(self):

        self.output_dir = "artifacts/reports"

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )


    def run(
        self,
        state: WorkflowState
    ) -> dict[str, Any]:

        print("\n" + "=" * 60)
        print("AUTOXLAB - REPORT GENERATION AGENT")
        print("=" * 60)


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
        # HTML REPORT
        # ====================================================

        html_path = os.path.join(
            self.output_dir,
            f"{run_id}_report.html"
        )


        metrics_html = ""

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float)
            ):

                metrics_html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value:.4f}</td>
                </tr>
                """


        html_content = f"""
<!DOCTYPE html>

<html>

<head>

    <title>AutoXLab Report</title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}

        h1 {{
            color: #174ea6;
        }}

        .card {{
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            background: #f5f7fa;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
        }}

        th {{
            background: #eeeeee;
        }}

    </style>

</head>

<body>

    <h1>AutoXLab Experiment Report</h1>


    <div class="card">

        <h2>Experiment Information</h2>

        <p>
            <b>Run ID:</b> {run_id}
        </p>

        <p>
            <b>Problem Type:</b> {problem_type}
        </p>

        <p>
            <b>Best Model:</b> {best_model}
        </p>

        <p>
            <b>Best Score:</b> {best_score:.4f}
        </p>

    </div>


    <div class="card">

        <h2>Model Metrics</h2>

        <table>

            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>

            {metrics_html}

        </table>

    </div>


    <div class="card">

        <h2>Recommendation</h2>

        <p>
            {recommendation}
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

            file.write(
                html_content
            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        pdf_path = os.path.join(
            self.output_dir,
            f"{run_id}_report.pdf"
        )


        try:

            from reportlab.lib.pagesizes import A4

            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle
            )

            from reportlab.lib import colors

            from reportlab.lib.styles import (
                getSampleStyleSheet
            )


            styles = getSampleStyleSheet()


            document = SimpleDocTemplate(
                pdf_path,
                pagesize=A4
            )


            elements = []


            elements.append(
                Paragraph(
                    "AutoXLab Experiment Report",
                    styles["Title"]
                )
            )


            elements.append(
                Spacer(1, 20)
            )


            elements.append(
                Paragraph(
                    f"Run ID: {run_id}",
                    styles["Normal"]
                )
            )


            elements.append(
                Paragraph(
                    f"Problem Type: {problem_type}",
                    styles["Normal"]
                )
            )


            elements.append(
                Paragraph(
                    f"Best Model: {best_model}",
                    styles["Normal"]
                )
            )


            elements.append(
                Paragraph(
                    f"Best Score: {best_score:.4f}",
                    styles["Normal"]
                )
            )


            elements.append(
                Spacer(1, 20)
            )


            elements.append(
                Paragraph(
                    "Model Metrics",
                    styles["Heading2"]
                )
            )


            table_data = [
                ["Metric", "Value"]
            ]


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


            elements.append(
                Paragraph(
                    "Recommendation",
                    styles["Heading2"]
                )
            )


            elements.append(
                Paragraph(
                    recommendation,
                    styles["Normal"]
                )
            )


            document.build(
                elements
            )


        except Exception as e:

            print(
                f"PDF generation warning: {e}"
            )

            pdf_path = ""


        # ====================================================
        # RETURN STATE
        # ====================================================

        return {

            "html_report_path":
                html_path,

            "report_path":
                pdf_path,

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