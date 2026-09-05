import pandas as pd
from typing import List, Dict, Tuple
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np

class TopicModeler:
    """BERTopic-based topic modeling and chapterization"""
    
    def __init__(self, df: pd.DataFrame, min_topic_size: int = 10):
        self.df = df.copy()
        self.min_topic_size = min_topic_size
        self.model = None
        self.topics = None
        
    def fit_topics(self, nr_topics: int = None) -> Tuple[BERTopic, List[int]]:
        """Extract topics using BERTopic"""
        # Prepare documents
        documents = self.df['message'].fillna('').tolist()
        
        # Initialize BERTopic with SentenceTransformers
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.model = BERTopic(
            embedding_model=sentence_model,
            min_topic_size=self.min_topic_size,
            nr_topics=nr_topics,
            language="english",
            calculate_probabilities=True
        )
        
        # Fit model
        self.topics, probs = self.model.fit_transform(documents)
        
        self.df['topic_id'] = self.topics
        self.df['topic_probability'] = [max(prob) for prob in probs]
        
        return self.model, self.topics
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get topic representations and frequencies"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_topics() first.")
        
        topic_info = self.model.get_topic_info()
        return topic_info
    
    def get_messages_by_topic(self, topic_id: int) -> pd.DataFrame:
        """Retrieve all messages for a specific topic"""
        return self.df[self.df['topic_id'] == topic_id]
    
    def chapterize_by_inactivity(self, gap_threshold_hours: int = 4) -> pd.DataFrame:
        """Split chat into chapters based on inactivity gaps"""
        self.df = self.df.sort_values('date')
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        
        # Calculate time gaps
        self.df['time_gap'] = self.df['datetime'].diff().dt.total_seconds() / 3600
        
        # Mark chapter boundaries
        self.df['new_chapter'] = (self.df['time_gap'] > gap_threshold_hours).astype(int)
        self.df['chapter_id'] = self.df['new_chapter'].cumsum()
        
        # Add chapter metadata
        chapter_info = self.df.groupby('chapter_id').agg({
            'datetime': ['min', 'max'],
            'sender': lambda x: x.value_counts().index[0],  # Most active sender
            'message': 'count'
        }).reset_index()
        
        chapter_info.columns = ['chapter_id', 'start_time', 'end_time', 
                               'dominant_sender', 'message_count']
        
        chapter_info['duration_hours'] = (
            chapter_info['end_time'] - chapter_info['start_time']
        ).dt.total_seconds() / 3600
        
        self.df = self.df.merge(chapter_info, on='chapter_id', how='left')
        
        return self.df
    
    def get_chapter_topics(self) -> pd.DataFrame:
        """Get dominant topic per chapter"""
        if 'chapter_id' not in self.df.columns:
            self.chapterize_by_inactivity()
        
        chapter_topics = self.df.groupby('chapter_id').agg({
            'topic_id': lambda x: x.value_counts().index[0] if len(x) > 0 else -1,
            'start_time': 'first',
            'end_time': 'first',
            'message_count': 'first'
        }).reset_index()
        
        # Merge with topic representations
        if self.model is not None:
            topic_info = self.model.get_topic_info()
            chapter_topics = chapter_topics.merge(
                topic_info[['Topic', 'Name', 'Representation']],
                left_on='topic_id',
                right_on='Topic',
                how='left'
            )
        
        return chapter_topics
