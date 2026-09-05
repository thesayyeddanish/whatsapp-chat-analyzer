import pandas as pd
from typing import Dict

class EmotionDetector:
    """Emotion detection - disabled on Streamlit Cloud (requires transformers)"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        raise NotImplementedError("Emotion detection requires transformers library which is too large for Streamlit Cloud free tier")
    
    def detect_emotions(self, batch_size: int = 8):
        """Not available"""
        pass
    
    def get_emotion_distribution(self):
        """Not available"""
        pass
