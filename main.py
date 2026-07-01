import os
import shutil
import joblib
import requests
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from feature_extraction import process_single_video


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "best_features.pkl"
#LIME_EXPLAINER_PATH = "lime_explainer.pkl"

GEMINI_API_KEY = "AIzaSyCNQJawqsEdYTRRYE3t807Twc5UOuL3wNg"

CLASS_NAMES = ["Advanced", "Intermediate"]

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)
best_features = joblib.load(FEATURES_PATH)
#lime_explainer = joblib.load(LIME_EXPLAINER_PATH)


def make_lime_explanation(X_selected: pd.DataFrame, prediction: int):

    return [

    ]


def make_gemini_feedback(level, confidence, features, lime_result, height=None, weight=None):
    prompt = f"""
        You are an AI climbing coach.
    
        The uploaded climbing video was analyzed using:
        1. MediaPipe pose estimation
        2. Machine learning classification
        3. Explainable AI interpretation
    
        Prediction:
        {level}
    
        Confidence:
        {confidence}
    
        User body information:
        Height: {height if height else "Not provided"} cm
        Weight: {weight if weight else "Not provided"} kg
    
        Selected movement feature values:
        {features}
    
        LIME explanation:
        {lime_result}
    
        Write coaching feedback in Korean.
    
        Important:
        Do not list the user's height or weight directly.
        Use the height and weight only to personalize the coaching advice.
        For example:
        - If the climber is relatively short, suggest using momentum, earlier foot placement, higher hip movement, or dynamic movement when appropriate.
        - If the climber is relatively tall, suggest using reach advantage, straighter arms, wider stance, and controlled body positioning when appropriate.
        - If the climber is relatively heavy, suggest efficient weight transfer, foot pressure, core tension, and reducing unnecessary upper-body pulling when appropriate.
        - If the climber is relatively light, suggest using body tension, stable feet, and controlled movement instead of relying only on flexibility or reach.
    
        Use this format:
        1. 분석 결과
        2. 움직임 특징
        3. 맞춤 코칭
        4. 다음 훈련 추천
    
        Keep it under 8 sentences.
        Do not use markdown.
        Do not use long explanations.
        Make it suitable for a mobile app screen.
        Make the explanation easy for an amateur climber to understand.
    """

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    response = requests.post(
        url,
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        },
        timeout=60,
    )

    data = response.json()

    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "Gemini 피드백 생성에 실패했습니다.")
    )


@app.get("/")
def home():
    return {"message": "AI Climbing Coach server is running."}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), height: str = Form(None), weight: str = Form(None)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df_summary = process_single_video(file_path)

    if df_summary is None or df_summary.empty:
        return {
            "success": False,
            "error": "Feature extraction failed."
        }

    X_selected = df_summary[best_features]

    prediction = int(model.predict(X_selected)[0])
    level = CLASS_NAMES[prediction]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X_selected).max())

    feature_values = X_selected.iloc[0].to_dict()

    lime_result = make_lime_explanation(
        X_selected=X_selected,
        prediction=prediction,
    )

    feedback = make_gemini_feedback(
        level=level,
        confidence=confidence,
        features=feature_values,
        lime_result=lime_result,
        height=height,
        weight=weight
    )

    return {
        "success": True,
        "level": level,
        "prediction": prediction,
        "confidence": confidence,
        "features": feature_values,
        "lime": lime_result,
        "feedback": feedback,
    }