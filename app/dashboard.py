import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
import pandas as pd

# Start Spark session
spark = SparkSession.builder.appName("StrokeDashboard").getOrCreate()

st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")

# Title
st.title("🩺 Real-Time Stroke Risk Prediction Dashboard")

# Load model
model = PipelineModel.load("models/stroke_pipeline")

# Load data — fallback to CSV if Parquet isn't found
try:
    df = spark.read.parquet("data/stroke_test_data.parquet")
except:
    df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)
    df = df.drop("id")

# Ensure correct dtypes
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast(FloatType()))

# Handle missing bmi
mean_bmi = df.select("bmi").agg({"bmi": "mean"}).collect()[0][0]
df = df.fillna({"bmi": mean_bmi})

# Apply model
predictions = model.transform(df)

# Select relevant columns
pred_df = predictions.select("age", "avg_glucose_level", "bmi", "prediction", "probability", "stroke") \
                     .toPandas()

# Extract probability
pred_df["stroke_risk"] = pred_df["probability"].apply(lambda x: x[1] if isinstance(x, list) else x.values[1])

# Risk Classification
def risk_level(prob):
    if prob > 0.8:
        return "🔴 High"
    elif prob > 0.5:
        return "🟠 Medium"
    else:
        return "🟢 Low"

pred_df["Risk Level"] = pred_df["stroke_risk"].apply(risk_level)

# Main display
st.subheader("📊 Prediction Results (sample)")
st.dataframe(pred_df[["age", "avg_glucose_level", "bmi", "stroke_risk", "Risk Level", "prediction"]].head(30), use_container_width=True)

# Risk Distribution
st.subheader("📈 Stroke Risk Distribution")
st.bar_chart(pred_df["stroke_risk"].value_counts(bins=10, sort=False))

# Footer
st.markdown("---")
st.markdown("✅ Built with PySpark + Streamlit | 🔍 Model: GBTClassifier")
