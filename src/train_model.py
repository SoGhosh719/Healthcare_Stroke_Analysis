from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("StrokePrediction").getOrCreate()

# Step 2: Load Data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)
df = df.drop("id")

# Step 3: Cast numeric columns explicitly to FloatType
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast(FloatType()))

# Step 4: Handle Missing Values
mean_bmi = df.select("bmi").agg({"bmi": "mean"}).collect()[0][0]
df = df.fillna({"bmi": mean_bmi})

# Step 5: Encode Categorical Features
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec") for col in categorical_cols]

# Step 6: Assemble Features
assembler_inputs = [col + "_vec" for col in categorical_cols] + ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_unscaled")
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")

# Step 7: Balance Classes (Manual Oversampling for Stroke Cases)
major_df = df.filter("stroke == 0")
minor_df = df.filter("stroke == 1")
ratio = int(major_df.count() / minor_df.count())
oversampled_df = major_df.unionAll(minor_df.sample(withReplacement=True, fraction=float(ratio), seed=42))

# Step 8: Train-Test Split
train_df, test_df = oversampled_df.randomSplit([0.8, 0.2], seed=42)

# Step 9: Define Model
gbt = GBTClassifier(labelCol="stroke", featuresCol="features", maxIter=20, maxDepth=5)

# Step 10: Create Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

# Step 11: Train Model
model = pipeline.fit(train_df)

# Step 12: Evaluate
predictions = model.transform(test_df)

evaluator = BinaryClassificationEvaluator(labelCol="stroke", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

f1_eval = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="f1")
precision_eval = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="precisionByLabel")
recall_eval = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="recallByLabel")

f1 = f1_eval.evaluate(predictions, {f1_eval.metricLabel: 1.0})
precision = precision_eval.evaluate(predictions, {precision_eval.metricLabel: 1.0})
recall = recall_eval.evaluate(predictions, {recall_eval.metricLabel: 1.0})

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision (stroke): {precision:.4f}")
print(f"Recall (stroke): {recall:.4f}")
print(f"F1 Score (stroke): {f1:.4f}")

# Step 13: Save Model
model.write().overwrite().save("models/stroke_pipeline")


# Step 13: Extract and Save Feature Importances
gbt_model = model.stages[-1]  # GBTClassifier is the last stage
importances = gbt_model.featureImportances.toArray()
feature_names = assembler_inputs

importance_data = list(zip(feature_names, importances))
importance_data_sorted = sorted(importance_data, key=lambda x: x[1], reverse=True)

# Print feature importances
print("\nTop Feature Importances:")
for name, score in importance_data_sorted:
    print(f"{name}: {score:.4f}")

# Optional: Save to CSV for visualization/dashboard
import pandas as pd
df_feat = pd.DataFrame(importance_data_sorted, columns=["Feature", "Importance"])
df_feat.to_csv("data/feature_importance.csv", index=False)

# Step 14: Save Predictions for Analysis
predictions.select("age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", 
                   "prediction", "probability", "stroke").toPandas().to_csv("data/predictions.csv", index=False)

predictions.select("stroke", "prediction", "probability") \
           .write.mode("overwrite") \
           .option("header", True) \
           .csv("data/prediction_output")
