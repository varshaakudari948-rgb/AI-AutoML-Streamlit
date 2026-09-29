import os
import json




from datetime import datetime
from typing import Any

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

from dotenv import load_dotenv

from graph.state import WorkflowState

from agents.memory.chroma_memory import (
    save_experiment_memory
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///autoxlab.db"
)


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    echo=False
)


# ============================================================
# SESSION
# ============================================================

SessionLocal = sessionmaker(
    bind=engine
)


# ============================================================
# BASE
# ============================================================

Base = declarative_base()


# ============================================================
# EXPERIMENT TABLE
# ============================================================

class ExperimentLog(Base):

    __tablename__ = "experiments"

    id = Column(
        Integer,
        primary_key=True
    )

    run_id = Column(
        String(100),
        index=True
    )

    target = Column(
        String(200)
    )

    problem_type = Column(
        String(50)
    )

    best_model = Column(
        String(200)
    )

    best_score = Column(
        Float
    )

    metrics = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# ============================================================
# DATABASE AGENT
# ============================================================

class DatabaseAgent:
    """
    AutoXLab Database Agent.

    Responsibilities:
    - Initialize database
    - Store experiment results
    - Store model information
    - Store evaluation metrics
    - Return database information to LangGraph state
    """

    def __init__(self):

        self.engine = engine
        self.SessionLocal = SessionLocal

        # Create database tables
        self.initialize_database()

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    def initialize_database(self):

        Base.metadata.create_all(
            self.engine
        )

        print(
            "[Database Agent] Database initialized."
        )

    # ========================================================
    # SAVE EXPERIMENT
    # ========================================================

    def log_experiment(
        self,
        state: WorkflowState
    ) -> int:

        session = self.SessionLocal()

        try:

            metrics = state.get(
                "metrics",
                {}
            )

            record = ExperimentLog(

                run_id=state.get(
                    "run_id",
                    "unknown"
                ),

                target=state.get(
                    "target_column",
                    "unknown"
                ),

                problem_type=state.get(
                    "problem_type",
                    "unknown"
                ),

                best_model=state.get(
                    "best_model_name",
                    "unknown"
                ),

                best_score=float(
                    state.get(
                        "best_score",
                        0.0
                    )
                ),

                metrics=json.dumps(
                    metrics
                )
            )

            session.add(
                record
            )

            session.commit()

            session.refresh(
                record
            )

            print(
                f"[Database Agent] "
                f"Experiment saved. ID: {record.id}"
            )

            return record.id

        except Exception:

            session.rollback()

            raise

        finally:

            session.close()

    # ========================================================
    # MAIN AGENT
    # ========================================================
    def run(self, state: WorkflowState) -> dict[str, Any]:

        print("\n" + "=" * 60)
        print("AUTOXLAB - DATABASE AGENT")
        print("=" * 60)

        # ----------------------------------------------------
        # VALIDATE REQUIRED FIELDS
        # ----------------------------------------------------

        required_fields = [
            "run_id",
            "target_column",
            "problem_type",
            "best_model_name",
            "best_score",
            "metrics"
        ]

        for field in required_fields:

            if field not in state:

                raise ValueError(
                    f"Required state field '{field}' is missing."
                )

        # ----------------------------------------------------
        # SAVE TO SQL DATABASE
        # ----------------------------------------------------

        experiment_id = self.log_experiment(
            state
        )

        print(
            f"[Database Agent] SQL experiment ID: "
            f"{experiment_id}"
        )

        # ----------------------------------------------------
        # SAVE TO CHROMA
        # ----------------------------------------------------

        try:

            save_experiment_memory(
                state
            )

            chroma_status = (
                "Experiment memory saved to ChromaDB."
            )

            print(
                "[Database Agent] "
                "Experiment saved to ChromaDB."
            )

        except Exception as e:

            chroma_status = (
                f"ChromaDB memory failed: {str(e)}"
            )

            print(
                "[Database Agent] "
                f"ChromaDB error: {str(e)}"
            )

        # ----------------------------------------------------
        # RETURN STATE
        # ----------------------------------------------------

        return {
            "status": "database_logging_completed",

            "messages": [
                "Experiment successfully stored in SQL database.",
                f"SQL experiment ID: {experiment_id}",
                chroma_status
            ],

            "error": ""
        }