from fastapi import FastAPI

from pydantic import BaseModel
class PredictionRequest(BaseModel):
    age: int
    salary: int
    

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    return{
        "age": request.age,
        "salary": request.salary,
    }

    