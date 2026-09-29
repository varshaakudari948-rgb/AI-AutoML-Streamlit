import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///autoxlab.db"
)

USE_CHROMA = os.getenv(
    "USE_CHROMA",
    "true"
).lower() == "true"

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "data/chroma"
)

MAX_RETRAINS = int(
    os.getenv(
        "MAX_RETRAINS",
        "2"
    )
)

MIN_ACCEPTABLE_SCORE = float(
    os.getenv(
        "MIN_ACCEPTABLE_SCORE",
        "0.80"
    )
)


print("AutoXLab Configuration")
print("-----------------------")
print("Database:", DATABASE_URL)
print("Chroma enabled:", USE_CHROMA)
print("Chroma path:", CHROMA_PATH)
print("Max retrains:", MAX_RETRAINS)
print("Minimum score:", MIN_ACCEPTABLE_SCORE)