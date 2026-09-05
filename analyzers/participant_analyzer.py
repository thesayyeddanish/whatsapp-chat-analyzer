import pandas as pd
from typing import Dict, List
from collections import Counter
import emoji

class ParticipantAnalyzer:
    """Analyze individual participant behavior and dynamics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_sender_leaderboard(self) -> pd.DataFrame:
        """Message count, word count, char count, avg length per user"""
        leaderboard = self.df.groupby('sender').agg({
            'message': 'count',
            'word_count': 'sum',
            'message_length': ['sum', 'mean'],
            'media_type': lambda x: (x == 'text').sum()
        }).reset_index()
        
        leaderboard.columns = [
            'sender', 'message_count', 'total_words', 
            'total_chars', 'avg_message_length', 'text_messages'
        ]
        
        leaderboard['avg_words_per_message'] = (
            leaderboard['total_words'] / leaderboard['message_count']
        )
        
        return leaderboard.sort_values('message_count', ascending=False)
    
    def calculate_double_texting_index(self) -> pd.DataFrame:
        """Who sends multiple consecutive messages before reply"""
        self.df = self.df.sort_values('date')
        
        double_text_counts = {}
        
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender].copy()
            sender_msgs['prev_sender'] = sender_msgs['sender'].shift(1)
            
            # Count consecutive messages by same sender
            consecutive = (sender_msgs['prev_sender'] == sender_msgs['sender']).sum()
            double_text_counts[sender] = consecutive
        
        result = pd.DataFrame(list(double_text_counts.items()), 
                            columns=['sender', 'consecutive_messages'])
        result['double_text_ratio'] = (
            result['consecutive_messages'] / 
            self.df.groupby('sender').size().reindex(result['sender']).values
        )
        
        return result.sort_values('consecutive_messages', ascending=False)
    
    def get_vocabulary_stats(self, top_n: int = 50) -> Dict:
        """Top words, emojis per user"""
        from nltk.corpus import stopwords
        import re
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        vocab_stats = {}
        
        for sender in self.df['sender'].unique():
            messages = self.df[self.df['sender'] == sender]['message'].tolist()
            text = ' '.join(messages).lower()
            
            # Remove URLs and special chars
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            
            # Tokenize
            words = re.findall(r'\b\w+\b', text)
            words = [w for w in words if w not in stop_words and len(w) > 2]
            
            # Top words
            word_freq = Counter(words).most_common(top_n)
            
            # Emoji extraction
            emojis = [c for c in ' '.join(messages) if c in emoji.EMOJI_DATA]
            emoji_freq = Counter(emojis).most_common(20)
            
            vocab_stats[sender] = {
                'top_words': word_freq,
                'top_emojis': emoji_freq,
                'unique_words': len(set(words)),
                'total_words': len(words)
            }
        
        return vocab_stats
    
    def get_media_ratios(self) -> pd.DataFrame:
        """Media and attachment ratios per user"""
        media_stats = self.df.groupby(['sender', 'media_type']).size().reset_index(name='count')
        
        # Pivot to get media type columns
        media_pivot = media_stats.pivot(index='sender', columns='media_type', values='count').fillna(0)
        
        # Calculate percentages
        total_per_sender = self.df.groupby('sender').size()
        for col in media_pivot.columns:
            media_pivot[f'{col}_pct'] = (
                media_pivot[col] / total_per_sender.reindex(media_pivot.index) * 100
            )
        
        return media_pivot
