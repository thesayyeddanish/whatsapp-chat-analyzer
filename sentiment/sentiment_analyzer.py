import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """VADER sentiment analysis - lightweight for Streamlit Cloud"""
    
    def __init__(self, df: pd.DataFrame, use_transformer: bool = False):
        self.df = df.copy()
        self.vader = SentimentIntensityAnalyzer()
        # Ignore transformer parameter - we only use VADER
    
    def analyze_vader(self) -> pd.DataFrame:
        """Apply VADER sentiment scoring to all messages"""
        scores = self.df['message'].fillna('').apply(lambda x: self.vader.polarity_scores(str(x)))
        scores_df = pd.DataFrame(scores.tolist(), index=self.df.index)
        
        # Add compound-based classification
        scores_df['vader_sentiment'] = scores_df['compound'].apply(
            lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
        )
        
        self.df = pd.concat([self.df, scores_df], axis=1)
        return self.df
    
    def get_mood_index(self, freq: str = 'W') -> pd.DataFrame:
        """Aggregate sentiment into weekly/monthly mood index"""
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        
        mood = self.df.groupby([
            pd.Grouper(key='datetime', freq=freq),
            'sender'
        ]).agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'message': 'count'
        }).reset_index()
        
        mood.columns = ['period', 'sender', 'avg_compound', 'avg_positive', 
                       'avg_negative', 'avg_neutral', 'message_count']
        
        # Classify mood
        mood['mood_category'] = mood['avg_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
        )
        
        return mood
