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
# SESSION STATE INITIALIZATION
# ============================================================

if "run_id" not in st.session_state:

    st.session_state["run_id"] = None


if "config" not in st.session_state:

    st.session_state["config"] = None


if "result" not in st.session_state:

    st.session_state["result"] = None


if "run_started" not in st.session_state:

    st.session_state["run_started"] = False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Experiment Configuration"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=[
        "csv",
        "xlsx",
        "xls"
    ]
)


llm_provider = st.sidebar.selectbox(
    "AI Reasoning Model",
    [
        "openai",
        "gemini",
        "claude"
    ]
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
    # DATASET NAME
    # --------------------------------------------------------

    uploaded_name = uploaded_file.name


    # --------------------------------------------------------
    # LOAD DATASET DIRECTLY FOR PREVIEW
    # --------------------------------------------------------

    try:

        if uploaded_name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"Could not read dataset: {str(e)}"
        )

        st.stop()


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(
        "📊 Dataset Preview"
    )

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

    st.subheader(
        "📋 Dataset Profile"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )


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

    st.subheader(
        "📈 Dataset Visualization"
    )


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

    run_experiment = st.button(
        "🚀 Run Autonomous Experiment",
        type="primary"
    )


    if run_experiment:

        # ----------------------------------------------------
        # CREATE A NEW RUN ID
        # ----------------------------------------------------

        run_id = str(
            uuid.uuid4()
        )

        st.session_state[
            "run_id"
        ] = run_id


        # ----------------------------------------------------
        # CREATE RUN DIRECTORY
        # ----------------------------------------------------

        run_dir = os.path.join(
            "runs",
            run_id
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # SAVE DATASET
        # ----------------------------------------------------

        input_path = os.path.join(
            run_dir,
            uploaded_name
        )


        try:

            with open(
                input_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        except Exception as e:

            st.error(
                f"Could not save dataset: "
                f"{str(e)}"
            )

            st.stop()


        # ----------------------------------------------------
        # INITIAL LANGGRAPH STATE
        # ----------------------------------------------------

        initial_state = {

            "dataset_path":
                input_path,

            "target_column":
                target,

            "run_id":
                run_id,

            "llm_provider":
                llm_provider,

            "use_pca":
                use_pca,

            "minimum_score":
                minimum_score,

            "max_retrains":
                int(
                    max_retrains
                ),

            "retrain_count":
                0,

            "experiment_number":
                1,

            "human_approved":
                False,

            "approval_status":
                "pending",

            "retrain_required":
                False,

            "messages":
                [],

            "status":
                "Starting..."
        }


        # ----------------------------------------------------
        # LANGGRAPH CHECKPOINT CONFIGURATION
        # ----------------------------------------------------

        config = {

            "configurable": {

                "thread_id":
                    run_id
            }
        }


        st.session_state[
            "config"
        ] = config


        st.session_state[
            "run_started"
        ] = True


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


                st.session_state[
                    "result"
                ] = result


                # ------------------------------------------------
                # CHECK WHETHER GRAPH IS WAITING FOR HUMAN
                # ------------------------------------------------

                interrupts = result.get(
                    "__interrupt__"
                )


                if interrupts:

                    st.info(
                        "✅ Experiment completed "
                        "and is waiting for human approval."
                    )

                else:

                    st.success(
                        "✅ Workflow execution completed."
                    )


                st.rerun()


            except Exception as e:

                st.error(
                    f"Workflow failed: {str(e)}"
                )

                st.exception(e)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.get(
    "result"
)


if result:

    # ========================================================
    # EXPERIMENT INFORMATION
    # ========================================================

    experiment_number = int(
        result.get(
            "experiment_number",
            1
        ) or 1
    )


    retrain_count = int(
        result.get(
            "retrain_count",
            0
        ) or 0
    )


    max_retrain_value = int(
        result.get(
            "max_retrains",
            max_retrains
        ) or max_retrains
    )


    # ========================================================
    # SHOW RETRAINING STATUS
    # ========================================================

    if experiment_number > 1:

        st.success(
            f"🔄 Retraining completed — "
            f"Experiment {experiment_number}"
        )

        st.write(
            f"Retraining loop: "
            f"**{retrain_count}/{max_retrain_value}**"
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    st.subheader(
        "🤖 AI Analysis"
    )


    llm_analysis = result.get(
        "llm_analysis",
        ""
    )


    if llm_analysis:

        st.info(
            llm_analysis
        )

    else:

        st.warning(
            "No AI analysis was generated."
        )


    # ========================================================
    # HUMAN APPROVAL / INTERRUPT
    # ========================================================

    interrupts = result.get(
        "__interrupt__"
    )


    if interrupts:

        interrupt_data = (
            interrupts[0].value
        )


        # ----------------------------------------------------
        # DISPLAY APPROVAL MESSAGE
        # ----------------------------------------------------

        st.warning(
            "⚠️ Human approval required"
        )


        st.subheader(
            "Model Approval"
        )


        # ----------------------------------------------------
        # EXPERIMENT NUMBER
        # ----------------------------------------------------

        interrupt_experiment = (
            interrupt_data.get(
                "experiment_number",
                experiment_number
            )
            if isinstance(
                interrupt_data,
                dict
            )
            else experiment_number
        )


        # ----------------------------------------------------
        # DISPLAY EXPERIMENT INFORMATION
        # ----------------------------------------------------

        st.write(
            f"**Experiment:** "
            f"{interrupt_experiment}"
        )


        st.json(
            interrupt_data
        )


        # ----------------------------------------------------
        # APPROVAL BUTTONS
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2
        )


        with col1:

            approve = st.button(
                "✅ Approve Model",
                key=(
                    f"approve_"
                    f"{experiment_number}"
                )
            )


        with col2:

            reject = st.button(
                "❌ Reject Model",
                key=(
                    f"reject_"
                    f"{experiment_number}"
                )
            )


        # ====================================================
        # HANDLE HUMAN DECISION
        # ====================================================

        if approve or reject:

            config = st.session_state.get(
                "config"
            )


            if config is None:

                st.error(
                    "Workflow configuration "
                    "is missing."
                )

                st.stop()


            # ------------------------------------------------
            # APPROVE
            # ------------------------------------------------

            if approve:

                approval_value = {

                    "approved":
                        True
                }


                spinner_text = (
                    "✅ Model approved. "
                    "Continuing workflow..."
                )


                status_message = (
                    "Model approved successfully."
                )


            # ------------------------------------------------
            # REJECT
            # ------------------------------------------------

            else:

                approval_value = {

                    "approved":
                        False
                }


                spinner_text = (
                    "🔄 Model rejected. "
                    "Retraining the model "
                    "and generating new results..."
                )


                status_message = (
                    "Model rejected. "
                    "Starting retraining..."
                )


            # ------------------------------------------------
            # RESUME GRAPH
            # ------------------------------------------------

            with st.spinner(
                spinner_text
            ):

                try:

                    final_result = graph.invoke(

                        Command(
                            resume=
                                approval_value
                        ),

                        config=config
                    )


                    # ------------------------------------------------
                    # SAVE NEW RESULT
                    # ------------------------------------------------

                    st.session_state[
                        "result"
                    ] = final_result


                    # ------------------------------------------------
                    # SHOW TEMPORARY STATUS
                    # ------------------------------------------------

                    if reject:

                        st.info(
                            status_message
                        )


                    else:

                        st.success(
                            status_message
                        )


                    # ------------------------------------------------
                    # IMPORTANT:
                    # RERUN SO NEW RESULT IS DISPLAYED
                    # ------------------------------------------------

                    st.rerun()


                except Exception as e:

                    st.error(
                        "Failed to resume workflow: "
                        f"{str(e)}"
                    )

                    st.exception(e)


        # ----------------------------------------------------
        # STOP HERE WHILE WAITING FOR HUMAN
        # ----------------------------------------------------

        st.stop()


    # ========================================================
    # EXPERIMENT RESULTS
    # ========================================================

    st.header(
        "🎯 Experiment Results"
    )


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(
        5
    )


    with col1:

        st.metric(
            "Experiment",
            experiment_number
        )


    with col2:

        st.metric(
            "Problem Type",
            result.get(
                "problem_type",
                "-"
            )
        )


    with col3:

        st.metric(
            "Best Model",
            result.get(
                "best_model_name",
                "-"
            )
        )


    with col4:

        best_score = result.get(
            "best_score",
            0
        )


        st.metric(
            "Best Score",
            f"{best_score:.4f}"
        )


    with col5:

        st.metric(
            "Retraining",
            f"{retrain_count}/"
            f"{max_retrain_value}"
        )


    # ========================================================
    # MODEL METRICS
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

                try:

                    display_value = (
                        f"{float(metric_value):.4f}"
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    display_value = str(
                        metric_value
                    )


                st.metric(
                    metric_name.upper(),
                    display_value
                )

    else:

        st.info(
            "No model metrics available."
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


    # --------------------------------------------------------
    # SHOW VISUALIZATION PATHS FROM OTHER AGENTS
    # --------------------------------------------------------

    if visualization_paths:

        for path in visualization_paths:

            if (
                path
                and
                os.path.exists(path)
            ):

                st.image(
                    path,
                    use_container_width=True
                )


    # --------------------------------------------------------
    # ALSO SHOW ML EVALUATION IMAGE
    # --------------------------------------------------------

    evaluation_image_path = result.get(
        "evaluation_image_path",
        ""
    )


    if (
        evaluation_image_path
        and
        os.path.exists(
            evaluation_image_path
        )
    ):

        st.image(
            evaluation_image_path,
            caption=(
                f"Evaluation - "
                f"Experiment "
                f"{experiment_number}"
            ),
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


    # ========================================================
    # HTML REPORT
    # ========================================================

    if (
        html_report
        and
        os.path.exists(
            html_report
        )
    ):

        with open(
            html_report,
            "rb"
        ) as file:

            st.download_button(
                "📊 Download HTML Report",
                file,
                file_name=(
                    "autoxlab_report.html"
                ),
                key="download_html_report"
            )


    # ========================================================
    # PDF REPORT
    # ========================================================

    if (
        pdf_report
        and
        os.path.exists(
            pdf_report
        )
    ):

        with open(
            pdf_report,
            "rb"
        ) as file:

            st.download_button(
                "📄 Download PDF Report",
                file,
                file_name=(
                    "autoxlab_report.pdf"
                ),
                key="download_pdf_report"
            )


    # ========================================================
    # NOTEBOOK
    # ========================================================

    if (
        notebook
        and
        os.path.exists(
            notebook
        )
    ):

        with open(
            notebook,
            "rb"
        ) as file:

            st.download_button(
                "📓 Download Notebook",
                file,
                file_name=(
                    "autoxlab_experiment.ipynb"
                ),
                key="download_notebook"
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

    human_approved = result.get(
        "human_approved"
    )


    if human_approved is True:

        st.success(
            "✅ Final model approved by human."
        )


    elif (
        human_approved is False
        and
        not interrupts
        and
        result.get("status") != "retraining"
    ):

        st.error(
            "❌ Final model was rejected."
        )


    # ========================================================
    # EXPERIMENT STATUS
    # ========================================================

    st.subheader(
        "📌 Experiment Status"
    )


    status = result.get(
        "status",
        "Unknown"
    )


    st.write(
        f"**Status:** {status}"
    )


    st.write(
        f"**Experiment number:** "
        f"{experiment_number}"
    )


    st.write(
        f"**Retraining loops used:** "
        f"{retrain_count}"
    )


    st.write(
        f"**Maximum retraining loops:** "
        f"{max_retrain_value}"
    )