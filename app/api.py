import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# -----------------------------
# Helpers: make Java/Spark work on Linux containers
# -----------------------------
def ensure_java():
    if os.environ.get("JAVA_HOME"):
        return

    java_path = shutil.which("java")
    if java_path:
        try:
            real_java = Path(
                subprocess.check_output(["readlink", "-f", java_path]).decode().strip()
            )
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


def extract_prob_1(prob_obj: Any) -> Optional[float]:
    # Handles DenseVector, SparseVector, list, numpy array
    if prob_obj is None:
        return None
    try:
        return float(prob_obj[1])
    except Exception:
        try:
            return float(prob_obj.values[1])
        except Exception:
            return None


# -----------------------------
# Input schema (match your dataset columns)
# -----------------------------
Gender = Literal["Male", "Female", "Other"]
YesNo = Literal["Yes", "No"]
WorkType = Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
ResidenceType = Literal["Urban", "Rural"]
SmokingStatus = Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


class StrokeFeatures(BaseModel):
    gender: Gender = "Male"
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(0, ge=0, le=1)
    heart_disease: int = Field(0, ge=0, le=1)
    ever_married: YesNo = "No"
    work_type: WorkType = "Private"
    Residence_type: ResidenceType = "Urban"
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(25.0, ge=0)  # default if not provided
    smoking_status: SmokingStatus = "Unknown"


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Stroke Prediction API", version="1.0.0")

# Optional but helpful if a frontend calls this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later (your domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spark = build_spark("StrokeAPI")
model = PipelineModel.load("models/stroke_pipeline")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: StrokeFeatures):
    """
    POST /predict

    Body example:
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
    try:
        data: Dict[str, Any] = payload.model_dump()

        # Ensure Spark gets exactly the expected columns
        # (and consistent types)
        row = {
            "gender": str(data["gender"]),
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "ever_married": str(data["ever_married"]),
            "work_type": str(data["work_type"]),
            "Residence_type": str(data["Residence_type"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": str(data["smoking_status"]),
        }

        df = spark.createDataFrame([row])

        out = model.transform(df).select("prediction", "probability").collect()[0]
        pred = int(out["prediction"])
        prob_1 = extract_prob_1(out["probability"])

        return {
            "stroke_prediction": pred,
            "stroke_risk_probability": prob_1,
            "input": row,  # useful for debugging client side
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(e).__name__}: {str(e)}"
        )
