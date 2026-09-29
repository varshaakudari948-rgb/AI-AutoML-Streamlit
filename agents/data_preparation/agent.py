from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graph.state import WorkflowState


class DataPreparationAgent:
    """
    AutoXLab Data Preparation Agent.

    Responsibilities:
        1. Load CSV / Excel datasets
        2. Validate dataset
        3. Detect target column
        4. Detect problem type
        5. Detect numeric and categorical columns
        6. Handle missing values
        7. Remove duplicate rows
        8. Detect numerical outliers
        9. Save cleaned dataset
        10. Return results compatible with WorkflowState
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls",
    }

    def __init__(
        self,
        output_directory: str = "data/processed",
    ) -> None:

        self.output_directory = Path(output_directory)

        # Create output directory if it doesn't exist
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # MAIN METHOD
    # ============================================================

    def run(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the complete data preparation pipeline.
        """

        dataset_path = state.get("dataset_path")

        if not dataset_path:
            raise ValueError(
                "dataset_path is missing from WorkflowState."
            )

        print("\n" + "=" * 60)
        print("AUTOXLAB - DATA PREPARATION AGENT")
        print("=" * 60)

        # --------------------------------------------------------
        # 1. Validate dataset path
        # --------------------------------------------------------

        path = self._validate_dataset_path(dataset_path)

        print(f"Dataset: {path}")

        # --------------------------------------------------------
        # 2. Load dataset
        # --------------------------------------------------------

        df = self._load_dataset(path)

        print(f"Original shape: {df.shape}")

        # --------------------------------------------------------
        # 3. Validate dataframe
        # --------------------------------------------------------

        self._validate_dataframe(df)

        # --------------------------------------------------------
        # 4. Store original information
        # --------------------------------------------------------

        original_columns = df.columns.tolist()

        dataset_shape = df.shape

        # --------------------------------------------------------
        # 5. Detect missing values BEFORE cleaning
        # --------------------------------------------------------

        missing_values_before = (
            df.isnull()
            .sum()
            .to_dict()
        )

        # --------------------------------------------------------
        # 6. Remove duplicate rows
        # --------------------------------------------------------

        duplicate_count = int(
            df.duplicated().sum()
        )

        if duplicate_count > 0:
            df = df.drop_duplicates()

        print(
            f"Duplicates removed: {duplicate_count}"
        )

        # --------------------------------------------------------
        # 7. Clean column names
        # --------------------------------------------------------

        df.columns = self._clean_column_names(
            df.columns
        )

        # --------------------------------------------------------
        # 8. Detect target column
        # --------------------------------------------------------

        target_column = self._detect_target_column(
            df,
            state,
        )

        print(
            f"Target column: {target_column}"
        )

        # --------------------------------------------------------
        # 9. Detect problem type
        # --------------------------------------------------------

        problem_type = self._detect_problem_type(
            df[target_column]
        )

        print(
            f"Problem type: {problem_type}"
        )

        # --------------------------------------------------------
        # 10. Detect column types
        # --------------------------------------------------------

        feature_columns = [
            column
            for column in df.columns
            if column != target_column
        ]

        numeric_columns = (
            df[feature_columns]
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            df[feature_columns]
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

        # --------------------------------------------------------
        # 11. Convert obvious numeric strings
        # --------------------------------------------------------

        df = self._convert_numeric_columns(
            df,
            feature_columns,
        )

        # Re-detect columns after conversion
        numeric_columns = (
            df[feature_columns]
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            df[feature_columns]
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ]
            )
            .columns
            .tolist()
        )

        # --------------------------------------------------------
        # 12. Detect outliers
        # --------------------------------------------------------

        outliers_detected = (
            self._detect_outliers(
                df,
                numeric_columns,
            )
        )

        print(
            f"Potential outliers detected: "
            f"{outliers_detected}"
        )

        # --------------------------------------------------------
        # 13. Handle missing values
        # --------------------------------------------------------

        df = self._handle_missing_values(
            df,
            numeric_columns,
            categorical_columns,
        )

        # --------------------------------------------------------
        # 14. Missing values AFTER cleaning
        # --------------------------------------------------------

        missing_values_after = (
            df.isnull()
            .sum()
            .to_dict()
        )

        # --------------------------------------------------------
        # 15. Save cleaned dataset
        # --------------------------------------------------------

        cleaned_path = (
            self.output_directory
            / "cleaned_dataset.csv"
        )

        df.to_csv(
            cleaned_path,
            index=False,
        )

        print(
            f"Cleaned dataset saved: "
            f"{cleaned_path}"
        )

        # --------------------------------------------------------
        # 16. Prepare messages
        # --------------------------------------------------------

        messages = [
            "Dataset loaded successfully.",
            f"Original shape: {dataset_shape}",
            f"Duplicates removed: {duplicate_count}",
            f"Target column: {target_column}",
            f"Problem type: {problem_type}",
            f"Numeric features: {len(numeric_columns)}",
            f"Categorical features: {len(categorical_columns)}",
            f"Outliers detected: {outliers_detected}",
            "Missing values handled.",
            "Cleaned dataset saved successfully.",
        ]

        # --------------------------------------------------------
        # 17. Return updated WorkflowState
        # --------------------------------------------------------

        return {
            "dataset_path": str(path),

            "cleaned_dataset_path": str(
                cleaned_path
            ),

            "target_column": target_column,

            "problem_type": problem_type,

            "dataset_shape": dataset_shape,

            "original_columns": original_columns,

            "feature_columns": feature_columns,

            "numeric_columns": numeric_columns,

            "categorical_columns": categorical_columns,

            "missing_values_before": (
                missing_values_before
            ),

            "missing_values_after": (
                missing_values_after
            ),

            "duplicates_removed": (
                duplicate_count
            ),

            "outliers_detected": (
                outliers_detected
            ),

            "status": (
                "data_preparation_completed"
            ),

            "messages": messages,

            "error": "",
        }

    # ============================================================
    # VALIDATE DATASET PATH
    # ============================================================

    def _validate_dataset_path(
        self,
        dataset_path: str,
    ) -> Path:

        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: "
                f"{', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        return path

    # ============================================================
    # LOAD DATASET
    # ============================================================

    def _load_dataset(
        self,
        path: Path,
    ) -> pd.DataFrame:

        extension = path.suffix.lower()

        if extension == ".csv":

            return pd.read_csv(path)

        if extension in {".xlsx", ".xls"}:

            return pd.read_excel(path)

        raise ValueError(
            f"Unsupported dataset format: {extension}"
        )

    # ============================================================
    # VALIDATE DATAFRAME
    # ============================================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset must contain at least "
                "one feature and one target column."
            )

        if df.shape[0] < 2:
            raise ValueError(
                "Dataset must contain at least "
                "two rows."
            )

    # ============================================================
    # CLEAN COLUMN NAMES
    # ============================================================

    def _clean_column_names(
        self,
        columns: pd.Index,
    ) -> list[str]:

        cleaned_columns = []

        for column in columns:

            name = str(column).strip()

            name = (
                name
                .replace(" ", "_")
                .replace("-", "_")
            )

            cleaned_columns.append(name)

        # Handle duplicate column names
        unique_columns = []
        counts: dict[str, int] = {}

        for column in cleaned_columns:

            if column not in counts:

                counts[column] = 0
                unique_columns.append(column)

            else:

                counts[column] += 1

                unique_columns.append(
                    f"{column}_{counts[column]}"
                )

        return unique_columns

    # ============================================================
    # DETECT TARGET COLUMN
    # ============================================================

    def _detect_target_column(
        self,
        df: pd.DataFrame,
        state: WorkflowState,
    ) -> str:

        # First priority:
        # Target explicitly supplied by user
        supplied_target = state.get(
            "target_column"
        )

        if supplied_target:

            if supplied_target in df.columns:

                return supplied_target

            # Try cleaned version
            cleaned_target = (
                supplied_target
                .strip()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if cleaned_target in df.columns:

                return cleaned_target

            raise ValueError(
                f"Specified target column "
                f"'{supplied_target}' "
                f"was not found."
            )

        # Common target names
        target_candidates = [
            "target",
            "label",
            "class",
            "y",
            "output",
            "response",
            "outcome",
            "prediction",
        ]

        lowercase_columns = {
            column.lower(): column
            for column in df.columns
        }

        for candidate in target_candidates:

            if candidate in lowercase_columns:

                return lowercase_columns[
                    candidate
                ]

        # If no obvious target is found,
        # use the final column.
        #
        # The website can later allow the user
        # to choose the target explicitly.

        return df.columns[-1]

    # ============================================================
    # DETECT PROBLEM TYPE
    # ============================================================

    def _detect_problem_type(
        self,
        target: pd.Series,
    ) -> str:

        # Remove missing values for analysis
        clean_target = target.dropna()

        if clean_target.empty:
            raise ValueError(
                "Target column contains no valid values."
            )

        # Numeric target
        if pd.api.types.is_numeric_dtype(
            clean_target
        ):

            unique_count = (
                clean_target.nunique()
            )

            sample_count = len(clean_target)

            # Binary / low-cardinality numeric target
            if unique_count <= 10:

                return "classification"

            # Integer target with low cardinality
            if (
                pd.api.types.is_integer_dtype(
                    clean_target
                )
                and unique_count <= 20
            ):

                return "classification"

            # Otherwise regression
            return "regression"

        # Non-numeric target → classification
        return "classification"

    # ============================================================
    # CONVERT NUMERIC STRINGS
    # ============================================================

    def _convert_numeric_columns(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:

        df = df.copy()

        for column in feature_columns:

            if (
                df[column].dtype
                == "object"
            ):

                converted = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                non_null_original = (
                    df[column].notna().sum()
                )

                non_null_converted = (
                    converted.notna().sum()
                )

                # Convert only when most
                # existing values are numeric
                if (
                    non_null_original > 0
                    and
                    non_null_converted
                    / non_null_original
                    >= 0.90
                ):

                    df[column] = converted

        return df

    # ============================================================
    # HANDLE MISSING VALUES
    # ============================================================

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> pd.DataFrame:

        df = df.copy()

        # Numeric columns → median
        for column in numeric_columns:

            if df[column].isnull().any():

                median_value = (
                    df[column].median()
                )

                df[column] = (
                    df[column]
                    .fillna(median_value)
                )

        # Categorical columns → mode
        for column in categorical_columns:

            if df[column].isnull().any():

                mode_values = (
                    df[column].mode()
                )

                if not mode_values.empty:

                    mode_value = mode_values.iloc[0]

                    df[column] = (
                        df[column]
                        .fillna(mode_value)
                    )

                else:

                    df[column] = (
                        df[column]
                        .fillna("Unknown")
                    )

        return df

    # ============================================================
    # OUTLIER DETECTION
    # ============================================================

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str],
    ) -> int:

        total_outliers = 0

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            # Constant column
            if iqr == 0:
                continue

            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )

            outliers = (
                (series < lower_bound)
                |
                (series > upper_bound)
            )

            total_outliers += int(
                outliers.sum()
            )

        return total_outliers


# ================================================================
# SIMPLE LOCAL TEST
# ================================================================

if __name__ == "__main__":

    print(
        "DataPreparationAgent loaded successfully."
    )

    agent = DataPreparationAgent()

    print(
        f"Output directory: "
        f"{agent.output_directory}"
    )