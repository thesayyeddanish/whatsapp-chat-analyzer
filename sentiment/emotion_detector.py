import pandas as pd
from typing import Dict, List
from transformers import pipeline

class EmotionDetector:
    """Fine-grained emotion profiling (Joy, Anger, Sadness, Fear, etc.)"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Load emotion detection model
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            top_k=None
        )
    
    def detect_emotions(self, batch_size: int = 8) -> pd.DataFrame:
        """Classify each message into emotions"""
        messages = self.df['message'].fillna('').tolist()
        
        all_emotions = []
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            try:
                results = self.emotion_classifier(batch)
                all_emotions.extend(results)
            except:
                # Fallback
                all_emotions.extend([[]] * len(batch))
        
        # Extract top emotion per message
        top_emotions = []
        emotion_scores = {
            'joy': [], 'sadness': [], 'anger': [], 
            'fear': [], 'surprise': [], 'love': []
        }
        
        for result in all_emotions:
            if result and len(result) > 0:
                # Get highest scoring emotion
                top = max(result, key=lambda x: x['score'])
                top_emotions.append(top['label'])
                
                # Store all emotion scores
                for emotion in emotion_scores.keys():
                    score = next((r['score'] for r in result if r['label'] == emotion), 0)
                    emotion_scores[emotion].append(score)
            else:
                top_emotions.append('neutral')
                for emotion in emotion_scores.keys():
                    emotion_scores[emotion].append(0)
        
        self.df['dominant_emotion'] = top_emotions
        
        # Add individual emotion score columns
        for emotion, scores in emotion_scores.items():
            self.df[f'emotion_{emotion}'] = scores
        
        return self.df
    
    def get_emotion_distribution(self) -> pd.DataFrame:
        """Emotion breakdown by participant"""
        emotion_dist = self.df.groupby(['sender', 'dominant_emotion']).size().reset_index(name='count')
        
        # Calculate percentages
        total_per_sender = self.df.groupby('sender').size()
        emotion_dist['percentage'] = (
            emotion_dist['count'] / 
            emotion_dist['sender'].map(total_per_sender) * 100
        )
        
        return emotion_dist
    
    def get_weekly_emotion_index(self) -> pd.DataFrame:
        """Weekly emotion intensity averages"""
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        self.df['week'] = self.df['datetime'].dt.isocalendar().week
        
        weekly_emotions = self.df.groupby(['week', 'sender']).agg({
            'emotion_joy': 'mean',
            'emotion_sadness': 'mean',
            'emotion_anger': 'mean',
            'emotion_fear': 'mean',
            'emotion_surprise': 'mean',
            'emotion_love': 'mean',
            'message': 'count'
        }).reset_index()
        
        return weekly_emotions
