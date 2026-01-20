import os
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# -----------------------------
# Helpers: make Java/Spark work on Streamlit Cloud / Linux containers
# -----------------------------
def ensure_java():
    # If already set, keep it
    if os.environ.get("JAVA_HOME"):
        return

    java_path = shutil.which("java")
    if java_path:
        try:
            real_java = Path(
                subprocess.check_output(["readlink", "-f", java_path]).decode().strip()
            )
            # .../bin/java -> JAVA_HOME is two levels up
            os.environ["JAVA_HOME"] = str(real_java.parent.parent)
            return
        except Exception:
            pass

    candidate = "/usr/lib/jvm/java-17-openjdk-amd64"
    if Path(candidate).exists():
        os.environ["JAVA_HOME"] = candidate


def build_spark(app_name: str) -> SparkSession:
    ensure_java()
    os.environ.setdefault("PYSPARK_PYTHON", "python3")
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")

    # Local Spark session + container-safe network binding
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Stroke Prediction API")

spark = build_spark("StrokeAPI")
model = PipelineModel.load("models/stroke_pipeline")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: dict):
    """
    Expects a JSON object with fields matching the pipeline's expected input columns.
    Example:
    {
      "gender": "Male",
      "age": 67,
      "hypertension": 0,
      "heart_disease": 1,
      "ever_married": "Yes",
      "work_type": "Private",
      "Residence_type": "Urban",
      "avg_glucose_level": 228.69,
      "bmi": 36.6,
      "smoking_status": "formerly smoked"
    }
    """
    # Create a single-row Spark DF
    df = spark.createDataFrame([data])

    # Run inference
    out = model.transform(df).select("prediction", "probability").collect()[0]
    pred = int(out["prediction"])

    # probability can be DenseVector; keep it JSON-friendly
    prob = out["probability"]
    try:
        prob_1 = float(prob[1])
    except Exception:
        try:
            prob_1 = float(prob.values[1])
        except Exception:
            prob_1 = None

    return {
        "stroke_prediction": pred,
        "stroke_risk_probability": prob_1,
    }
