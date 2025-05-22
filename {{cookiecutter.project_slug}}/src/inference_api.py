from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="LLM 뉴스 분류 API")

# 입력 데이터 스키마 정의
class InferenceInput(BaseModel):
    text: str

# 라벨 정의 (추론 대상 클래스 순서와 일치해야 함)
LABELS =['education', 'human interest', 'society', 'sport', 'crime, law and justice',
'disaster, accident and emergency incident', 'arts, culture, entertainment and media', 'politics',
'economy, business and finance', 'lifestyle and leisure', 'science and technology',
'health', 'labour', 'religion', 'weather', 'environment', 'conflict, war and peace']
# 모델 및 토크나이저 로드
MODEL_PATH = "classla/multilingual-IPTC-news-topic-classifier"  # 학습된 모델 경로 또는 HuggingFace 모델명
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 예측 API
@app.post("/predict")
def predict(input_data: InferenceInput):
    try:
        inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = round(probs[0][pred_idx].item(), 4)

        return {
            "label": LABELS[pred_idx],
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"🔥 Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

# 헬스 체크
@app.get("/health")
def health_check():
    return {"status": "ok"}