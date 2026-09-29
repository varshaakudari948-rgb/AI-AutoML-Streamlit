import os
import joblib
import numpy as np
import pandas as pd

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression
)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from graph.state import WorkflowState


class FeatureEngineeringAgent:
    """
    AutoXLab Feature Engineering Agent.

    Responsibilities:
    - Load cleaned dataset
    - Separate features and target
    - Detect numerical/categorical features
    - Impute missing values
    - Scale numerical features
    - Encode categorical features
    - Split train/test data
    - Perform statistical feature selection
    - Optionally perform PCA
    - Save preprocessing artifacts
    - Return information to LangGraph state
    """

    def __init__(self):
        pass

    # ============================================================
    # MAIN METHOD
    # ============================================================

    def run(
        self,
        state: WorkflowState
    ) -> dict[str, Any]:

        print("\n" + "=" * 60)
        print("AUTOXLAB - FEATURE ENGINEERING AGENT")
        print("=" * 60)

        # --------------------------------------------------------
        # 1. GET INPUTS FROM STATE
        # --------------------------------------------------------

        cleaned_dataset_path = state.get(
            "cleaned_dataset_path"
        )

        target = state.get(
            "target_column"
        )

        problem_type = state.get(
            "problem_type"
        )

        if not cleaned_dataset_path:
            raise ValueError(
                "cleaned_dataset_path is missing "
                "from WorkflowState."
            )

        if not target:
            raise ValueError(
                "target_column is missing "
                "from WorkflowState."
            )

        if not problem_type:
            raise ValueError(
                "problem_type is missing "
                "from WorkflowState."
            )

        # --------------------------------------------------------
        # 2. LOAD CLEANED DATASET
        # --------------------------------------------------------

        df = pd.read_csv(
            cleaned_dataset_path
        )

        print(
            f"Cleaned dataset shape: {df.shape}"
        )

        # --------------------------------------------------------
        # 3. SEPARATE X AND Y
        # --------------------------------------------------------

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' "
                f"not found in dataset."
            )

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        # --------------------------------------------------------
        # 4. DETECT COLUMN TYPES
        # --------------------------------------------------------

        numeric_columns = (
            X.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool"
                ]
            )
            .columns
            .tolist()
        )

        print(
            f"Numeric columns: "
            f"{numeric_columns}"
        )

        print(
            f"Categorical columns: "
            f"{categorical_columns}"
        )

        # --------------------------------------------------------
        # 5. NUMERICAL PIPELINE
        # --------------------------------------------------------

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # --------------------------------------------------------
        # 6. CATEGORICAL PIPELINE
        # --------------------------------------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        # --------------------------------------------------------
        # 7. CREATE TRANSFORMERS
        # --------------------------------------------------------

        transformers = []

        if numeric_columns:

            transformers.append(
                (
                    "numeric",
                    numerical_pipeline,
                    numeric_columns
                )
            )

        if categorical_columns:

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            )

        if not transformers:
            raise ValueError(
                "No usable feature columns "
                "were found."
            )

        # --------------------------------------------------------
        # 8. COLUMN TRANSFORMER
        # --------------------------------------------------------

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        # --------------------------------------------------------
        # 9. TRAIN / TEST SPLIT
        # --------------------------------------------------------

        stratify = None

        if problem_type == "classification":

            # Stratification is possible only when
            # every class has at least two samples.

            if y.value_counts().min() >= 2:

                stratify = y

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=stratify
            )
        )

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Testing samples: {len(X_test)}"
        )

        # --------------------------------------------------------
        # 10. FIT PREPROCESSOR
        # --------------------------------------------------------

        X_train_processed = (
            preprocessor.fit_transform(
                X_train,
                y_train
            )
        )

        X_test_processed = (
            preprocessor.transform(
                X_test
            )
        )

        # --------------------------------------------------------
        # 11. GET FEATURE NAMES
        # --------------------------------------------------------

        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

        print(
            f"Transformed features: "
            f"{len(feature_names)}"
        )

        # --------------------------------------------------------
        # 12. SELECT FEATURE SCORING FUNCTION
        # --------------------------------------------------------

        if problem_type == "classification":

            score_function = f_classif

        elif problem_type == "regression":

            score_function = f_regression

        else:

            raise ValueError(
                f"Unsupported problem type: "
                f"{problem_type}"
            )

        # --------------------------------------------------------
        # 13. CALCULATE FEATURE SCORES
        # --------------------------------------------------------

        scoring_selector = SelectKBest(
            score_func=score_function,
            k="all"
        )

        scoring_selector.fit(
            X_train_processed,
            y_train
        )

        scores = (
            scoring_selector.scores_
        )

        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------------
        # 14. SELECT TOP FEATURES
        # --------------------------------------------------------

        k = min(
            30,
            X_train_processed.shape[1]
        )

        selector = SelectKBest(
            score_func=score_function,
            k=k
        )

        X_train_selected = (
            selector.fit_transform(
                X_train_processed,
                y_train
            )
        )

        X_test_selected = (
            selector.transform(
                X_test_processed
            )
        )

        # --------------------------------------------------------
        # 15. GET SELECTED FEATURES
        # --------------------------------------------------------

        mask = (
            selector.get_support()
        )

        selected_features = [
            feature_names[i]
            for i, selected
            in enumerate(mask)
            if selected
        ]

        print(
            f"Selected features: "
            f"{len(selected_features)}"
        )

        # --------------------------------------------------------
        # 16. OPTIONAL PCA
        # --------------------------------------------------------

        use_pca = state.get(
            "use_pca",
            False
        )

        explained_variance = 0.0

        if (
            use_pca
            and
            X_train_selected.shape[1] > 2
        ):

            n_components = min(
                10,
                X_train_selected.shape[1],
                X_train_selected.shape[0]
            )

            pca = PCA(
                n_components=n_components,
                random_state=42
            )

            X_train_final = (
                pca.fit_transform(
                    X_train_selected
                )
            )

            X_test_final = (
                pca.transform(
                    X_test_selected
                )
            )

            explained_variance = float(
                pca
                .explained_variance_ratio_
                .sum()
            )

            final_features = [
                f"PC{i + 1}"
                for i in range(
                    n_components
                )
            ]

            print(
                f"PCA components: "
                f"{n_components}"
            )

            print(
                f"Explained variance: "
                f"{explained_variance:.4f}"
            )

        else:

            pca = None

            X_train_final = (
                X_train_selected
            )

            X_test_final = (
                X_test_selected
            )

            final_features = (
                selected_features
            )

        # --------------------------------------------------------
        # 17. CREATE ARTIFACT DIRECTORY
        # --------------------------------------------------------

        run_dir = os.path.dirname(
            cleaned_dataset_path
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        # --------------------------------------------------------
        # 18. ARTIFACT PATHS
        # --------------------------------------------------------

        preprocessor_path = os.path.join(
            run_dir,
            "preprocessor.joblib"
        )

        selector_path = os.path.join(
            run_dir,
            "selector.joblib"
        )

        pca_path = os.path.join(
            run_dir,
            "pca.joblib"
        )

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

        # --------------------------------------------------------
        # 19. SAVE PREPROCESSOR
        # --------------------------------------------------------

        joblib.dump(
            preprocessor,
            preprocessor_path
        )

        # --------------------------------------------------------
        # 20. SAVE FEATURE SELECTOR
        # --------------------------------------------------------

        joblib.dump(
            selector,
            selector_path
        )

        # --------------------------------------------------------
        # 21. SAVE PCA
        # --------------------------------------------------------

        joblib.dump(
            pca,
            pca_path
        )

        # --------------------------------------------------------
        # 22. SAVE TRAIN / TEST FEATURES
        # --------------------------------------------------------

        np.save(
            X_train_path,
            X_train_final
        )

        np.save(
            X_test_path,
            X_test_final
        )

        # --------------------------------------------------------
        # 23. SAVE TRAIN / TEST TARGETS
        # --------------------------------------------------------

        np.save(
            y_train_path,
            y_train.to_numpy()
        )

        np.save(
            y_test_path,
            y_test.to_numpy()
        )

        # --------------------------------------------------------
        # 24. CREATE FEATURE SCORES DICTIONARY
        # --------------------------------------------------------

        feature_scores = {
            feature_names[i]: float(
                scores[i]
            )
            for i in range(
                len(feature_names)
            )
        }

        # --------------------------------------------------------
        # 25. MESSAGES
        # --------------------------------------------------------

        messages = [
            "Numerical features scaled.",
            "Categorical features encoded.",
            "Training and testing datasets created.",
            "Features scored using statistical tests.",
            "Top features selected.",
        ]

        if pca is not None:

            messages.append(
                "PCA dimensionality reduction applied."
            )

        else:

            messages.append(
                "PCA dimensionality reduction skipped."
            )

        messages.append(
            "Feature engineering artifacts saved."
        )

        # --------------------------------------------------------
        # 26. RETURN LANGGRAPH STATE
        # --------------------------------------------------------

        return {

            "X_train_path": X_train_path,
            "X_test_path": X_test_path,
            "y_train_path": y_train_path,
            "y_test_path": y_test_path,

            "preprocessor_path":
                preprocessor_path,

            "feature_matrix_path":
                X_train_path,

            "target_vector_path":
                y_train_path,

            "selected_features":
                selected_features,

            "all_transformed_features":
                final_features,

            "feature_scores":
                feature_scores,

            "scaling_applied":
                bool(numeric_columns),

            "encoding_applied":
                bool(categorical_columns),

            "dimensionality_reduction":
                pca is not None,

            "explained_variance":
                explained_variance,

            "status":
                "feature_engineering_completed",

            "messages":
                messages,

            "error":
                "",
        }