from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

app = FastAPI()
spark = SparkSession.builder.appName("StrokeAPI").getOrCreate()
model = PipelineModel.load("models/stroke_pipeline")

@app.post("/predict")
async def predict(data: dict):
    df = spark.createDataFrame([data])
    result = model.transform(df).select("prediction").collect()[0][0]
    return {"stroke_prediction": int(result)}
