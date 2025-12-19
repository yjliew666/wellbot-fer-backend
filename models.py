from pydantic import BaseModel

class EmotionResult(BaseModel):
    emotion: str
    confidence: float


