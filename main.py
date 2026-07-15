import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field

from feature_extraction import process_single_video


# =========================================================
# Environment
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)
SUPABASE_VIDEO_BUCKET = os.getenv(
    "SUPABASE_VIDEO_BUCKET",
    "climbing-videos",
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="AI Climbing Coach API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Files and models
# =========================================================

UPLOAD_DIR = Path("uploads")

MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "best_features.pkl"
LIME_TRAINING_DATA_PATH = "lime_training_data.pkl"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(MODEL_PATH)
best_features = joblib.load(FEATURES_PATH)
lime_training_data = joblib.load(
    LIME_TRAINING_DATA_PATH
)

CLASS_NAMES = [
    "Advanced",
    "Intermediate",
]

FEATURE_NAME_MAP = {
    "shoulder_y_symmetry_mean":
        "Shoulder vertical symmetry",
    "hip_center_velocity_mean":
        "Hip center velocity",
    "left_shoulder_vertical_sway":
        "Left shoulder vertical sway",
    "hip_center_jerk_sd":
        "Hip center jerk standard deviation",
    "shoulder_center_velocity_mean":
        "Shoulder center velocity",
    "hip_center_velocity_sd":
        "Hip center velocity standard deviation",
    "right_hip_velocity_mean":
        "Right hip velocity",
    "hip_center_acceleration_sd":
        "Hip center acceleration standard deviation",
}

lime_explainer = LimeTabularExplainer(
    training_data=lime_training_data.values,
    feature_names=best_features,
    class_names=CLASS_NAMES,
    mode="classification",
    random_state=42,
)


# =========================================================
# Request models
# =========================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    height: str | None = None
    weight: str | None = None
    history: list[dict[str, Any]] | None = None
    profile: dict[str, Any] | None = None


class AnalyzeRequest(BaseModel):
    video_path: str = Field(min_length=1)
    height: str | None = None
    weight: str | None = None
    question: str | None = None


# =========================================================
# Supabase authentication and Storage
# =========================================================

def check_supabase_configuration() -> None:
    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured."
        )

    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY is not configured."
        )


def extract_bearer_token(
    authorization: str | None,
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing.",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header.",
        )

    return token.strip()


def get_authenticated_user_id(
    authorization: str | None,
) -> str:
    """
    Supabase access token을 확인하고 로그인 사용자의 UUID를 반환합니다.
    """
    check_supabase_configuration()

    access_token = extract_bearer_token(
        authorization
    )

    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
        },
        timeout=30,
    )

    if response.status_code != 200:
        print(
            "Supabase authentication error:",
            response.status_code,
            response.text,
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "Your session is invalid or has expired. "
                "Please sign in again."
            ),
        )

    try:
        user_data = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Supabase returned an invalid "
                "authentication response."
            ),
        ) from error

    user_id = user_data.get("id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="The authenticated user could not be identified.",
        )

    return str(user_id)


def validate_user_video_path(
    video_path: str,
    user_id: str,
) -> str:
    """
    사용자가 자신의 Storage 폴더에 있는 영상만 분석하도록 제한합니다.

    허용 형식:
    사용자UUID/파일명.mp4
    """
    normalized_path = video_path.strip().lstrip("/")

    expected_prefix = f"{user_id}/"

    if not normalized_path.startswith(expected_prefix):
        raise HTTPException(
            status_code=403,
            detail=(
                "You do not have permission "
                "to access this video."
            ),
        )

    if ".." in Path(normalized_path).parts:
        raise HTTPException(
            status_code=400,
            detail="The video path is invalid.",
        )

    return normalized_path


def download_video_from_supabase(
    video_path: str,
    destination: Path,
) -> None:
    """
    Private Supabase Storage 영상 파일을 Cloud Run 임시 폴더로
    다운로드합니다.
    """
    check_supabase_configuration()

    encoded_path = quote(
        video_path,
        safe="/",
    )

    encoded_bucket = quote(
        SUPABASE_VIDEO_BUCKET,
        safe="",
    )

    download_url = (
        f"{SUPABASE_URL}/storage/v1/object/"
        f"{encoded_bucket}/{encoded_path}"
    )

    try:
        response = requests.get(
            download_url,
            headers={
                "Authorization": (
                    f"Bearer "
                    f"{SUPABASE_SERVICE_ROLE_KEY}"
                ),
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
            },
            stream=True,
            timeout=300,
        )
    except requests.Timeout as error:
        raise RuntimeError(
            "The video download timed out."
        ) from error
    except requests.RequestException as error:
        raise RuntimeError(
            "Unable to connect to Supabase Storage."
        ) from error

    if response.status_code != 200:
        print(
            "Supabase Storage download error:",
            response.status_code,
            response.text,
        )

        raise RuntimeError(
            "Unable to download the video from storage."
        )

    with destination.open("wb") as output_file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                output_file.write(chunk)

    if (
        not destination.exists()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            "The downloaded video file is empty."
        )


# =========================================================
# LIME
# =========================================================

def replace_feature_name(
    feature_condition: str,
) -> str:
    for raw_name, readable_name in (
        FEATURE_NAME_MAP.items()
    ):
        if raw_name in feature_condition:
            return feature_condition.replace(
                raw_name,
                readable_name,
            )

    return feature_condition


def make_lime_explanation(
    X_selected: pd.DataFrame,
    prediction: int,
) -> list[dict[str, Any]]:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The model does not support probability prediction."
        )

    explanation = (
        lime_explainer.explain_instance(
            X_selected.iloc[0].values,
            model.predict_proba,
            num_features=len(best_features),
            labels=(prediction,),
        )
    )

    return [
        {
            "feature_condition":
                replace_feature_name(
                    feature_condition
                ),
            "importance":
                round(float(importance), 3),
        }
        for feature_condition, importance
        in explanation.as_list(
            label=prediction
        )
    ]


# =========================================================
# Gemini
# =========================================================

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return (
            "The Gemini API key is not configured. "
            "Please check the server environment variables."
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        "gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        response = requests.post(
            url,
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ]
            },
            timeout=60,
        )
    except requests.Timeout:
        return (
            "The AI response timed out. "
            "Please try again in a moment."
        )
    except requests.RequestException:
        return (
            "The AI service could not be reached. "
            "Please try again later."
        )

    try:
        data = response.json()
    except ValueError:
        return (
            "The AI service returned "
            "an invalid response."
        )

    if response.status_code == 429:
        return (
            "The AI usage limit has been reached. "
            "Please try again later."
        )

    if response.status_code == 503:
        return (
            "The AI service is temporarily busy. "
            "Please try again shortly."
        )

    if response.status_code != 200:
        print(
            "Gemini API error:",
            response.status_code,
            data,
        )

        return (
            "The AI service encountered an error. "
            "Please try again later."
        )

    candidates = data.get(
        "candidates",
        [],
    )

    if not candidates:
        print(
            "Gemini response without candidates:",
            data,
        )

        return (
            "The AI service did not return "
            "a response."
        )

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    if not parts:
        print(
            "Gemini response without parts:",
            data,
        )

        return (
            "The AI service returned "
            "an invalid response."
        )

    return parts[0].get(
        "text",
        "The AI service returned an empty response.",
    )


def make_gemini_feedback(
    level: str,
    confidence: float | None,
    lime_result: list[dict[str, Any]],
    height: str | None = None,
    weight: str | None = None,
    question: str | None = None,
) -> str:
    prompt = f"""
You are an AI climbing coach.

The uploaded climbing video was analyzed using:
1. MediaPipe pose estimation
2. Machine learning classification
3. LIME explainable AI

Prediction:
{level}

Confidence:
{
    confidence
    if confidence is not None
    else "Not available"
}

LIME explanation:
{lime_result}

User body information:
Height: {
    height
    if height
    else "Not provided"
} cm

Weight: {
    weight
    if weight
    else "Not provided"
} kg

User question:
{
    question
    if question
    else "No specific question was provided."
}

Write practical coaching feedback in English.

Important:
- Base the feedback mainly on the LIME explanation.
- Use the user's question when relevant.
- Do not list raw feature values.
- Do not directly list the user's height or weight.
- Use body information only to personalize advice.
- Do not make medical claims.
- Keep the response under 5 sentences.
- Do not use markdown.
- Do not provide long explanations.
- Make the response suitable for a mobile app screen.

Use this structure:
1. Analysis result
2. Main contributing factor
3. Personalized coaching advice
4. Recommended next training step
"""

    return call_gemini(prompt)


# =========================================================
# Routes
# =========================================================

@app.get("/")
def home():
    return {
        "message":
            "AI Climbing Coach server is running.",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "supabase_configured": bool(
            SUPABASE_URL
            and SUPABASE_SERVICE_ROLE_KEY
        ),
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
    }


@app.post("/chat")
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(
        default=None
    ),
):
    # 로그인 토큰 검증
    get_authenticated_user_id(
        authorization
    )

    recent_history = (
        request.history[-5:]
        if request.history
        else []
    )

    analysis_summary = []

    for item in recent_history:
        top_lime = (
            item.get("lime", [])[:3]
            if isinstance(item, dict)
            else []
        )

        raw_confidence = (
            item.get("confidence")
            if isinstance(item, dict)
            else None
        )

        try:
            confidence = (
                round(float(raw_confidence), 2)
                if raw_confidence is not None
                else None
            )
        except (TypeError, ValueError):
            confidence = None

        analysis_summary.append(
            {
                "level": (
                    item.get("level")
                    if isinstance(item, dict)
                    else None
                ),
                "confidence": confidence,
                "top_lime": [
                    lime.get(
                        "feature_condition"
                    )
                    for lime in top_lime
                    if isinstance(lime, dict)
                ],
            }
        )

    prompt = f"""
You are a friendly AI climbing coach.

Answer in English.

Use the user's profile and recent climbing analysis
history to provide personalized coaching.

User profile:
{
    request.profile
    if request.profile
    else "Not provided"
}

Recent climbing analysis summary:
{
    analysis_summary
    if analysis_summary
    else "No previous analysis history"
}

Current user question:
{request.message}

Important:
- Do not list raw feature values.
- Do not directly list height or weight.
- Use profile information only when relevant.
- If previous analyses exist, compare patterns over time.
- Use repeated LIME factors to explain movement patterns.
- If the user asks about progress, compare recent levels,
  confidence values, and repeated factors.
- Keep the answer practical, friendly, and under 5 sentences.
- Do not use markdown.
- Make the answer suitable for a mobile app screen.
"""

    answer = call_gemini(prompt)

    return {
        "success": True,
        "answer": answer,
    }


@app.post("/analyze")
async def analyze_video(
    request: AnalyzeRequest,
    authorization: str | None = Header(
        default=None
    ),
):
    user_id = get_authenticated_user_id(
        authorization
    )

    video_path = validate_user_video_path(
        video_path=request.video_path,
        user_id=user_id,
    )

    file_suffix = (
        Path(video_path)
        .suffix
        .lower()
    )

    if not file_suffix:
        file_suffix = ".mp4"

    allowed_suffixes = {
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
        ".avi",
    }

    if file_suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Please upload an MP4, MOV, M4V, "
                "WEBM, or AVI file."
            ),
        )

    unique_filename = (
        f"{uuid.uuid4().hex}"
        f"{file_suffix}"
    )

    file_path = (
        UPLOAD_DIR
        / unique_filename
    )

    try:
        # 1. Supabase Storage → Cloud Run 임시 저장
        download_video_from_supabase(
            video_path=video_path,
            destination=file_path,
        )

        # 2. 특징 추출
        df_summary = process_single_video(
            str(file_path)
        )

        if (
            df_summary is None
            or df_summary.empty
        ):
            return {
                "success": False,
                "error": (
                    "Video feature extraction failed. "
                    "Please make sure the full climber "
                    "is visible throughout the video."
                ),
            }

        missing_features = [
            feature
            for feature in best_features
            if feature not in df_summary.columns
        ]

        if missing_features:
            print(
                "Missing model features:",
                missing_features,
            )

            return {
                "success": False,
                "error": (
                    "The extracted video data is incomplete. "
                    "Please try another video."
                ),
            }

        # 3. 머신러닝 예측
        X_selected = df_summary[
            best_features
        ]

        prediction = int(
            model.predict(
                X_selected
            )[0]
        )

        if (
            prediction < 0
            or prediction >= len(
                CLASS_NAMES
            )
        ):
            raise RuntimeError(
                "The model returned an invalid prediction."
            )

        level = CLASS_NAMES[
            prediction
        ]

        confidence: float | None = None

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities = (
                model.predict_proba(
                    X_selected
                )[0]
            )

            confidence = float(
                probabilities[
                    prediction
                ]
            )

        # 4. LIME
        lime_result = (
            make_lime_explanation(
                X_selected=X_selected,
                prediction=prediction,
            )
        )

        # 5. Gemini
        feedback = make_gemini_feedback(
            level=level,
            confidence=confidence,
            lime_result=lime_result,
            height=request.height,
            weight=request.weight,
            question=request.question,
        )

        return {
            "success": True,
            "level": level,
            "prediction": prediction,
            "confidence": confidence,
            "lime": lime_result,
            "video_path": video_path,
            "height": request.height,
            "weight": request.weight,
            "question": request.question,
            "feedback": feedback,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Video analysis error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "during video analysis."
            ),
        ) from error

    finally:
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as cleanup_error:
            print(
                "Temporary file cleanup error:",
                cleanup_error,
            )