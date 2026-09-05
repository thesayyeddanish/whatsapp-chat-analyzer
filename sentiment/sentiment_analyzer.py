import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """VADER and Transformer-based sentiment analysis"""
    
    def __init__(self, df: pd.DataFrame, use_transformer: bool = True):
        self.df = df.copy()
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.roberta_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
    
    def analyze_vader(self) -> pd.DataFrame:
        """Apply VADER sentiment scoring to all messages"""
        scores = self.df['message'].apply(lambda x: self.vader.polarity_scores(str(x)))
        scores_df = pd.DataFrame(scores.tolist(), index=self.df.index)
        
        # Add compound-based classification
        scores_df['vader_sentiment'] = scores_df['compound'].apply(
            lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
        )
        
        self.df = pd.concat([self.df, scores_df], axis=1)
        return self.df
    
    def analyze_roberta(self, batch_size: int = 8) -> pd.DataFrame:
        """Apply RoBERTa transformer for fine-grained sentiment"""
        if not self.use_transformer:
            return self.df
        
        messages = self.df['message'].fillna('').tolist()
        
        # Process in batches to avoid memory issues
        all_results = []
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            try:
                results = self.roberta_classifier(batch)
                all_results.extend(results)
            except Exception as e:
                # Fallback for problematic messages
                all_results.extend([{'label': 'LABEL_1', 'score': 0.5}] * len(batch))
        
        # Parse results
        sentiment_labels = []
        confidence_scores = []
        
        for result in all_results:
            if isinstance(result, list):
                # Multi-class output
                best = max(result, key=lambda x: x['score'])
                sentiment_labels.append(best['label'])
                confidence_scores.append(best['score'])
            else:
                sentiment_labels.append(result['label'])
                confidence_scores.append(result['score'])
        
        self.df['roberta_sentiment'] = sentiment_labels
        self.df['roberta_confidence'] = confidence_scores
        
        # Map labels to readable format
        label_map = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        self.df['roberta_sentiment'] = self.df['roberta_sentiment'].map(label_map)
        
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
    
    def get_emotional_trajectory(self) -> pd.DataFrame:
        """Track sentiment changes over time"""
        self.df = self.df.sort_values('date')
        
        # Rolling average of compound score
        self.df['rolling_sentiment'] = self.df['compound'].rolling(window=10, min_periods=1).mean()
        
        # Detect sentiment shifts
        self.df['sentiment_shift'] = self.df['compound'].diff()
        
        return self.df
