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
    model_input = [[request.age, request.salary]]
    prediction = model.predict(model_input)
    return{
        "age": request.age,
        "salary": request.salary,
        "predicted_credit_limit": prediction[0]
    }

    