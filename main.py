import os
import shutil
import joblib
import requests
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lime.lime_tabular import LimeTabularExplainer

from feature_extraction import process_single_video

load_dotenv()

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
LIME_TRAINING_DATA_PATH = "lime_training_data.pkl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CLASS_NAMES = ["Advanced", "Intermediate"]

FEATURE_NAME_MAP = {
    "shoulder_y_symmetry_mean": "Shoulder vertical symmetry",
    "hip_center_velocity_mean": "Hip center velocity",
    "left_shoulder_vertical_sway": "Left shoulder vertical sway",
    "hip_center_jerk_sd": "Hip center jerk standard deviation",
    "shoulder_center_velocity_mean": "Shoulder center velocity",
    "hip_center_velocity_sd": "Hip center velocity standard deviation",
    "right_hip_velocity_mean": "Right hip velocity",
    "hip_center_acceleration_sd": "Hip center acceleration standard deviation",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)
best_features = joblib.load(FEATURES_PATH)
lime_training_data = joblib.load(LIME_TRAINING_DATA_PATH)

lime_explainer = LimeTabularExplainer(
    training_data=lime_training_data.values,
    feature_names=best_features,
    class_names=CLASS_NAMES,
    mode="classification",
    random_state=42,
)


class ChatRequest(BaseModel):
    message: str
    height: str | None = None
    weight: str | None = None
    history: list[dict] | None = None
    profile: dict | None = None


def replace_feature_name(feature_condition: str) -> str:
    for raw_name, readable_name in FEATURE_NAME_MAP.items():
        if raw_name in feature_condition:
            return feature_condition.replace(raw_name, readable_name)
    return feature_condition


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API Key가 설정되지 않았습니다. .env 파일을 확인해주세요."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    response = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )

    data = response.json()

    if response.status_code != 200:
        return f"Gemini 오류: {data}"

    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", f"Gemini 응답 형식 오류: {data}")
    )


def make_lime_explanation(X_selected: pd.DataFrame, prediction: int):
    exp = lime_explainer.explain_instance(
        X_selected.iloc[0].values,
        model.predict_proba,
        num_features=len(best_features),
        labels=(prediction,),
    )

    return [
        {
            "feature_condition": replace_feature_name(feature_condition),
            "importance": round(float(importance), 3),
        }
        for feature_condition, importance in exp.as_list(label=prediction)
    ]


def make_gemini_feedback(level, confidence, lime_result, height=None, weight=None):
    prompt = f"""
You are an AI climbing coach.

The uploaded climbing video was analyzed using:
1. MediaPipe pose estimation
2. Machine learning classification
3. LIME explainable AI

Prediction:
{level}

Confidence:
{confidence}

LIME explanation:
{lime_result}

User body information:
Height: {height if height else "Not provided"} cm
Weight: {weight if weight else "Not provided"} kg

Write coaching feedback in Korean based mainly on the LIME explanation.

Important:
Do not list raw feature values.
Do not list the user's height or weight directly.
Use height and weight only to personalize the coaching advice.
If the climber is relatively short, suggest momentum, earlier foot placement, higher hip movement, or dynamic movement when appropriate.
If the climber is relatively tall, suggest reach advantage, straighter arms, wider stance, and controlled body positioning when appropriate.
If the climber is relatively heavy, suggest efficient weight transfer, foot pressure, core tension, and reducing unnecessary upper-body pulling when appropriate.
If the climber is relatively light, suggest body tension, stable feet, and controlled movement.

Use this format:
1. 분석 결과
2. 주요 원인
3. 맞춤 코칭
4. 다음 훈련 추천

Keep it under 8 sentences.
Do not use markdown.
Do not use long explanations.
Make it suitable for a mobile app screen.
"""

    return call_gemini(prompt)


@app.get("/")
def home():
    return {"message": "AI Climbing Coach server is running."}


@app.post("/chat")
async def chat(request: ChatRequest):
    recent_history = request.history[-5:] if request.history else []

    prompt = f"""
You are a friendly AI climbing coach.

Answer in Korean.
Use the user's body information, profile, and previous climbing analysis history to give personalized coaching.

User body information:
Height: {request.height if request.height else "Not provided"} cm
Weight: {request.weight if request.weight else "Not provided"} kg

User profile:
{request.profile if request.profile else "Not provided"}

Recent climbing analysis history:
{recent_history if recent_history else "No previous analysis history"}

Important:
Do not list height or weight directly.
Use body information only when it helps personalize advice.
If previous analyses exist, compare patterns over time.
Use LIME results to explain recurring movement issues.
If the user asks about progress, compare recent predictions, confidence, and repeated LIME factors.
Keep the answer short, practical, and friendly.
Do not use markdown.

User question:
{request.message}
"""

    answer = call_gemini(prompt)

    return {
        "success": True,
        "answer": answer,
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    height: str = Form(None),
    weight: str = Form(None),
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df_summary = process_single_video(file_path)

    if df_summary is None or df_summary.empty:
        return {
            "success": False,
            "error": "Feature extraction failed.",
        }

    X_selected = df_summary[best_features]

    prediction = int(model.predict(X_selected)[0])
    level = CLASS_NAMES[prediction]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X_selected).max())

    lime_result = make_lime_explanation(
        X_selected=X_selected,
        prediction=prediction,
    )

    feedback = make_gemini_feedback(
        level=level,
        confidence=confidence,
        lime_result=lime_result,
        height=height,
        weight=weight,
    )

    return {
        "success": True,
        "level": level,
        "prediction": prediction,
        "confidence": confidence,
        "lime": lime_result,
        "height": height,
        "weight": weight,
        "feedback": feedback,
    }