from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from joblib import load
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
model = load(os.path.join(BASE_DIR, "model.pkl"))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def get_prediction(
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    bloodPressure: float = Form(...),
    skinThickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetesPedigreeFunction: float = Form(...),
    age: float = Form(...),
):
    data = np.array([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]])
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    probability_of_diabetes = probability[0][1]
    return JSONResponse(content={
        "prediction": int(prediction[0]),
        "probability": float(probability_of_diabetes),
    })