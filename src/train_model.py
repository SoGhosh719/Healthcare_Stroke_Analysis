from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("StrokePrediction").getOrCreate()

# Step 2: Load Data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)
df = df.drop("id")

# Step 3: Handle Missing Values
df = df.fillna({"bmi": df.select("bmi").agg({"bmi": "mean"}).collect()[0][0]})

# Step 4: Encode Categorical Features
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec") for col in categorical_cols]

# Step 5: Assemble Features
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
assembler_inputs = [col + "_vec" for col in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_unscaled")
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")

# Step 6: Balance Classes (Manual Oversampling for Stroke Cases)
major_df = df.filter("stroke == 0")
minor_df = df.filter("stroke == 1")
ratio = int(major_df.count() / minor_df.count())
oversampled_df = major_df.unionAll(minor_df.sample(withReplacement=True, fraction=ratio))

# Step 7: Train-Test Split
train_df, test_df = oversampled_df.randomSplit([0.8, 0.2], seed=42)

# Step 8: Define Model
gbt = GBTClassifier(labelCol="stroke", featuresCol="features", maxIter=20, maxDepth=5)

# Step 9: Create Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

# Step 10: Train Model
model = pipeline.fit(train_df)

# Step 11: Evaluate
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

# Step 12: Save Model
model.write().overwrite().save("models/stroke_pipeline")
