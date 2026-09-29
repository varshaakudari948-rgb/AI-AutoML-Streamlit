import os
import uuid

import pandas as pd
import streamlit as st

from langgraph.types import Command

from graph.workflow import graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AutoXLab",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
    }

    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AutoXLab</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Autonomous Multi-Agent Data Science Experiment Platform
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Experiment Configuration")


uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)


use_pca = st.sidebar.checkbox(
    "Enable dimensionality reduction",
    value=False
)


minimum_score = st.sidebar.slider(
    "Minimum acceptable score",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01
)


max_retrains = st.sidebar.number_input(
    "Maximum retraining loops",
    min_value=0,
    max_value=5,
    value=2
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    # --------------------------------------------------------
    # CREATE RUN ID
    # --------------------------------------------------------

    run_id = str(uuid.uuid4())

    run_dir = os.path.join(
        "runs",
        run_id
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE UPLOADED DATASET
    # --------------------------------------------------------

    input_path = os.path.join(
        run_dir,
        uploaded_file.name
    )

    with open(input_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(input_path)

    else:

        df = pd.read_excel(input_path)


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.write(
        f"Dataset shape: "
        f"**{df.shape[0]} rows × "
        f"{df.shape[1]} columns**"
    )


    # ========================================================
    # TARGET COLUMN
    # ========================================================

    target = st.selectbox(
        "Select target column",
        df.columns.tolist()
    )


    # ========================================================
    # DATASET PROFILE
    # ========================================================

    st.subheader("📋 Dataset Profile")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            )
        )


    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                df.duplicated()
                .sum()
            )
        )


    # ========================================================
    # DATASET VISUALIZATION
    # ========================================================

    st.subheader("📈 Dataset Visualization")


    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    if numeric_columns:

        selected_column = st.selectbox(
            "Select numeric feature",
            numeric_columns
        )


        chart_data = df[
            selected_column
        ].dropna()


        st.line_chart(
            chart_data
        )


    # ========================================================
    # RUN WORKFLOW
    # ========================================================

    if st.button(
        "🚀 Run Autonomous Experiment",
        type="primary"
    ):

        # ----------------------------------------------------
        # LANGGRAPH INITIAL STATE
        # ----------------------------------------------------

        initial_state = {

            "dataset_path":
                input_path,

            "target_column":
                target,

            "run_id":
                run_id,

            "use_pca":
                use_pca,

            "minimum_score":
                minimum_score,

            "max_retrains":
                int(max_retrains),

            "retrain_count":
                0,

            "experiment_number":
                1,

            "messages":
                [],

            "status":
                "Starting..."
        }


        # ----------------------------------------------------
        # LANGGRAPH CHECKPOINT CONFIG
        # ----------------------------------------------------

        config = {

            "configurable": {

                "thread_id":
                    run_id
            }
        }


        # ----------------------------------------------------
        # RUN GRAPH
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Autonomous agents are working..."
        ):

            try:

                result = graph.invoke(
                    initial_state,
                    config=config
                )


                # Save state for next Streamlit rerun

                st.session_state["run_id"] = run_id

                st.session_state["config"] = config

                st.session_state["result"] = result


                st.success(
                    "Workflow execution completed."
                )

            except Exception as e:

                st.error(
                    f"Workflow failed: {str(e)}"
                )

                st.exception(e)


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]


    # ========================================================
    # HUMAN APPROVAL / INTERRUPT
    # ========================================================

    interrupts = result.get(
        "__interrupt__"
    )


    if interrupts:

        interrupt_data = interrupts[
            0
        ].value


        st.warning(
            "⚠️ Human approval required"
        )


        st.subheader(
            "Model Approval"
        )


        st.json(
            interrupt_data
        )


        col1, col2 = st.columns(2)


        with col1:

            approve = st.button(
                "✅ Approve Model"
            )


        with col2:

            reject = st.button(
                "❌ Reject Model"
            )


        if approve or reject:

            approved = approve


            config = st.session_state[
                "config"
            ]


            with st.spinner(
                "Resuming workflow..."
            ):

                try:

                    final_result = graph.invoke(

                        Command(
                            resume={
                                "approved":
                                    approved
                            }
                        ),

                        config=config
                    )


                    st.session_state[
                        "result"
                    ] = final_result


                    st.rerun()


                except Exception as e:

                    st.error(
                        f"Failed to resume workflow: {str(e)}"
                    )

                    st.exception(e)


        st.stop()


    # ========================================================
    # EXPERIMENT RESULTS
    # ========================================================

    st.header(
        "🎯 Experiment Results"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Problem Type",
            result.get(
                "problem_type",
                "-"
            )
        )


    with col2:

        st.metric(
            "Best Model",
            result.get(
                "best_model_name",
                "-"
            )
        )


    with col3:

        best_score = result.get(
            "best_score",
            0
        )

        st.metric(
            "Best Score",
            f"{best_score:.4f}"
        )


    with col4:

        st.metric(
            "Retraining",
            result.get(
                "retrain_count",
                0
            )
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.subheader(
        "📊 Model Metrics"
    )


    metrics = result.get(
        "metrics",
        {}
    )


    if metrics:

        metric_columns = st.columns(
            len(metrics)
        )


        for index, (
            metric_name,
            metric_value
        ) in enumerate(
            metrics.items()
        ):

            with metric_columns[index]:

                st.metric(
                    metric_name.upper(),
                    f"{metric_value:.4f}"
                )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader(
        "🔬 Model Comparison"
    )


    model_results = result.get(
        "model_results",
        []
    )


    if model_results:

        model_df = pd.DataFrame(
            model_results
        )


        st.dataframe(
            model_df,
            use_container_width=True
        )

    else:

        st.info(
            "No model comparison results available."
        )


    # ========================================================
    # SELECTED FEATURES
    # ========================================================

    st.subheader(
        "🧩 Selected Features"
    )


    selected_features = result.get(
        "selected_features",
        []
    )


    if selected_features:

        st.write(
            selected_features
        )

    else:

        st.info(
            "No selected features available."
        )


    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    st.subheader(
        "📈 Model Visualizations"
    )


    visualization_paths = result.get(
        "visualization_paths",
        []
    )


    for path in visualization_paths:

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )


    # ========================================================
    # REPORTS
    # ========================================================

    st.subheader(
        "📁 Generated Artifacts"
    )


    html_report = result.get(
        "html_report_path"
    )


    pdf_report = result.get(
        "report_path"
    )


    notebook = result.get(
        "notebook_path"
    )


    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    if (
        html_report
        and
        os.path.exists(html_report)
    ):

        with open(
            html_report,
            "rb"
        ) as file:

            st.download_button(
                "📊 Download HTML Report",
                file,
                file_name="autoxlab_report.html"
            )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if (
        pdf_report
        and
        os.path.exists(pdf_report)
    ):

        with open(
            pdf_report,
            "rb"
        ) as file:

            st.download_button(
                "📄 Download PDF Report",
                file,
                file_name="autoxlab_report.pdf"
            )


    # --------------------------------------------------------
    # NOTEBOOK
    # --------------------------------------------------------

    if (
        notebook
        and
        os.path.exists(notebook)
    ):

        with open(
            notebook,
            "rb"
        ) as file:

            st.download_button(
                "📓 Download Notebook",
                file,
                file_name="autoxlab_experiment.ipynb"
            )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Agent Recommendation"
    )


    st.info(
        result.get(
            "recommendation",
            "No recommendation."
        )
    )


    # ========================================================
    # HUMAN APPROVAL STATUS
    # ========================================================

    if result.get(
        "human_approved"
    ) is True:

        st.success(
            "✅ Final model approved by human."
        )


    elif result.get(
        "human_approved"
    ) is False:

        st.error(
            "❌ Final model was rejected."
        )