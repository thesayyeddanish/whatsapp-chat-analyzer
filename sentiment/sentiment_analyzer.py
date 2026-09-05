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
        try:
            # Apply VADER to each message
            def get_vader_scores(text):
                if text is None or str(text).strip() == '':
                    return {'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'compound': 0.0}
                return self.vader.polarity_scores(str(text))
            
            scores = self.df['message'].apply(get_vader_scores)
            scores_df = pd.DataFrame(scores.tolist(), index=self.df.index)
            
            # Rename columns to match expected names
            scores_df = scores_df.rename(columns={
                'pos': 'positive',
                'neu': 'neutral',
                'neg': 'negative',
                'compound': 'compound'
            })
            
            # Add compound-based classification
            scores_df['vader_sentiment'] = scores_df['compound'].apply(
                lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
            )
            
            # Concatenate with original DataFrame
            self.df = pd.concat([self.df, scores_df], axis=1)
            
            return self.df
            
        except Exception as e:
            print(f"VADER sentiment analysis failed: {str(e)}")
            # Add default columns if analysis fails
            self.df['positive'] = 0.0
            self.df['neutral'] = 1.0
            self.df['negative'] = 0.0
            self.df['compound'] = 0.0
            self.df['vader_sentiment'] = 'neutral'
            return self.df
    
    def get_mood_index(self, freq: str = 'W') -> pd.DataFrame:
        """Aggregate sentiment into weekly/monthly mood index"""
        # Check if sentiment columns exist
        required_cols = ['compound', 'positive', 'negative', 'neutral']
        if not all(col in self.df.columns for col in required_cols):
            # Run sentiment analysis if not done
            self.analyze_vader()
        
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
