import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType


# -----------------------------
# Helpers: make Java/Spark work on Streamlit Cloud
# -----------------------------
def ensure_java():
    # If already set, keep it
    if os.environ.get("JAVA_HOME"):
        return

    # If java exists, infer JAVA_HOME from it
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

    # Common Debian/Ubuntu path
    candidate = "/usr/lib/jvm/java-17-openjdk-amd64"
    if Path(candidate).exists():
        os.environ["JAVA_HOME"] = candidate


@st.cache_resource
def get_spark():
    ensure_java()

    # These help in container environments
    os.environ.setdefault("PYSPARK_PYTHON", "python3")
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")

    # Create Spark session (local mode) + container-safe network settings
    spark = (
        SparkSession.builder
        .appName("StrokeDashboard")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    return spark


@st.cache_resource
def get_model():
    # IMPORTANT: path is relative to repo root at runtime
    return PipelineModel.load("models/stroke_pipeline")


@st.cache_data
def load_data(spark: SparkSession):
    # Load data — fallback to CSV if Parquet isn't found
    try:
        df = spark.read.parquet("data/stroke_test_data.parquet")
    except Exception:
        df = spark.read.csv(
            "data/healthcare-dataset-stroke-data.csv",
            header=True,
            inferSchema=True,
        )
        if "id" in df.columns:
            df = df.drop("id")

    # Ensure correct dtypes
    numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(FloatType()))

    # Handle missing bmi (safe even if bmi is all null)
    if "bmi" in df.columns:
        mean_bmi = df.selectExpr("avg(bmi) as mean_bmi").collect()[0]["mean_bmi"]
        if mean_bmi is not None:
            df = df.fillna({"bmi": float(mean_bmi)})

    return df


def extract_prob_second(x):
    """
    probability can be:
    - pyspark.ml.linalg.DenseVector / SparseVector
    - list
    - numpy array
    """
    try:
        return float(x[1])
    except Exception:
        try:
            return float(x.values[1])
        except Exception:
            return None


def risk_level(prob):
    if prob is None:
        return "⚪ Unknown"
    if prob > 0.8:
        return "🔴 High"
    elif prob > 0.5:
        return "🟠 Medium"
    else:
        return "🟢 Low"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")
st.title("🩺 Real-Time Stroke Risk Prediction Dashboard")

spark = get_spark()
model = get_model()
df = load_data(spark)

# Apply model
predictions = model.transform(df)

# Convert ONLY what you need to pandas (avoid huge memory use)
N = st.sidebar.slider("Rows to display", min_value=50, max_value=5000, value=500, step=50)

selected = predictions.select(
    "age", "avg_glucose_level", "bmi", "prediction", "probability", "stroke"
).limit(N)

pred_df = selected.toPandas()

# Extract probability for class 1
pred_df["stroke_risk"] = pred_df["probability"].apply(extract_prob_second)
pred_df["Risk Level"] = pred_df["stroke_risk"].apply(risk_level)

st.subheader("📊 Prediction Results (sample)")
st.dataframe(
    pred_df[["age", "avg_glucose_level", "bmi", "stroke_risk", "Risk Level", "prediction"]]
    .sort_values("stroke_risk", ascending=False)
    .head(30),
    use_container_width=True,
)

st.subheader("📈 Stroke Risk Distribution")
hist = pred_df["stroke_risk"].dropna().value_counts(bins=10, sort=False)
st.bar_chart(hist)

st.markdown("---")
st.markdown("✅ Built with PySpark + Streamlit | 🔍 Model: GBTClassifier")
