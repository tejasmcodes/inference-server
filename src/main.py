from fastapi import FastAPI
import joblib

from pydantic import BaseModel
class PredictionRequest(BaseModel):
    age: int
    salary: int
    

app = FastAPI()
model = joblib.load('models/regression_model.joblib')
print("Model loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    return{
        "age": request.age,
        "salary": request.salary,
    }

    