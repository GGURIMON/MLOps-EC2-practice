# inference.py
import pandas as pd
from fastapi import FastAPI
from app.schemas import PredictIn, PredictOut
import joblib


def get_model():
    
    model = joblib.load('model_pipeline.joblib')
    return model


MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.model_dump()])
    pred = MODEL.predict(df).item()
    return PredictOut(iris_class=pred)