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


@st.cache_resource
def get_spark():
    ensure_java()

    os.environ.setdefault("PYSPARK_PYTHON", "python3")
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")

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
    return PipelineModel.load("models/stroke_pipeline")


# ❗ DO NOT cache Spark DataFrames with st.cache_data
def load_data(spark: SparkSession):
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

    numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(FloatType()))

    if "bmi" in df.columns:
        mean_bmi = df.selectExpr("avg(bmi) as mean_bmi").collect()[0]["mean_bmi"]
        if mean_bmi is not None:
            df = df.fillna({"bmi": float(mean_bmi)})

    return df


def extract_prob_second(x):
    if x is None:
        return None
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

st.sidebar.header("Controls")
N = st.sidebar.slider("Rows to display", min_value=50, max_value=5000, value=500, step=50)

spark = get_spark()
model = get_model()

df = load_data(spark)
predictions = model.transform(df)

wanted_cols = ["age", "avg_glucose_level", "bmi", "prediction", "probability"]
if "stroke" in predictions.columns:
    wanted_cols.append("stroke")

selected = predictions.select(*wanted_cols).limit(N)
pred_df = selected.toPandas()

pred_df["stroke_risk"] = pred_df["probability"].apply(extract_prob_second)
pred_df["Risk Level"] = pred_df["stroke_risk"].apply(risk_level)

st.subheader("📊 Prediction Results (sample)")

table_cols = [
    c for c in ["age", "avg_glucose_level", "bmi", "stroke_risk", "Risk Level", "prediction", "stroke"]
    if c in pred_df.columns
]

pred_df_sorted = pred_df.copy()
pred_df_sorted["stroke_risk_sort"] = pd.to_numeric(pred_df_sorted["stroke_risk"], errors="coerce")

st.dataframe(
    pred_df_sorted[table_cols + ["stroke_risk_sort"]]
    .sort_values("stroke_risk_sort", ascending=False, na_position="last")
    .drop(columns=["stroke_risk_sort"])
    .head(30),
    use_container_width=True,
)

st.subheader("📈 Stroke Risk Distribution")
risk_series = pd.to_numeric(pred_df["stroke_risk"], errors="coerce").dropna()

if len(risk_series) == 0:
    st.info("No valid probability values found to plot.")
else:
    hist = risk_series.value_counts(bins=10, sort=False)
    st.bar_chart(hist)

st.markdown("---")
st.markdown("✅ Built with PySpark + Streamlit | 🔍 Model: GBTClassifier")
