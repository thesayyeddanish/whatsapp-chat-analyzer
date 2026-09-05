import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class NarrativeGenerator:
    """Generate readable story format from chat data"""
    
    def __init__(self, df: pd.DataFrame, chapter_df: pd.DataFrame = None):
        self.df = df
        self.chapter_df = chapter_df
        
    def generate_week_titles(self, title_model: str = 'tfidf') -> pd.DataFrame:
        """Generate weekly headlines using TF-IDF or lightweight LLM"""
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        self.df['year_week'] = self.df['datetime'].dt.strftime('%Y-W%W')
        
        week_summaries = []
        
        for week, group in self.df.groupby('year_week'):
            messages = group['message'].fillna('').tolist()
            
            # Simple TF-IDF approach for headline generation
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if len(messages) > 0:
                vectorizer = TfidfVectorizer(
                    max_features=5,
                    stop_words='english',
                    ngram_range=(2, 3)
                )
                
                try:
                    tfidf = vectorizer.fit_transform(messages)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get top phrases
                    top_phrases = feature_names[:3] if len(feature_names) >= 3 else feature_names
                    
                    headline = " | ".join(top_phrases).title()
                except:
                    headline = f"Week {week}"
            else:
                headline = f"Week {week}"
            
            week_summaries.append({
                'year_week': week,
                'headline': headline,
                'message_count': len(group),
                'participants': ', '.join(group['sender'].unique())
            })
        
        return pd.DataFrame(week_summaries)
    
    def generate_chapter_narrative(self, chapter_id: int) -> str:
        """Generate prose narrative for a chapter"""
        chapter_msgs = self.df[self.df['chapter_id'] == chapter_id].sort_values('date')
        
        if len(chapter_msgs) == 0:
            return "No messages in this chapter."
        
        # Extract key information
        start_date = chapter_msgs['date'].min()
        end_date = chapter_msgs['date'].max()
        participants = chapter_msgs['sender'].unique().tolist()
        
        # Get dominant topics/emotions
        if 'dominant_emotion' in chapter_msgs.columns:
            top_emotion = chapter_msgs['dominant_emotion'].mode().iloc[0] if len(chapter_msgs) > 0 else 'neutral'
        else:
            top_emotion = 'unknown'
        
        # Generate narrative
        narrative = f"""
## Chapter {chapter_id}: {start_date} to {end_date}

**Participants:** {', '.join(participants)}  
**Primary Emotion:** {top_emotion}  
**Messages:** {len(chapter_msgs)}

### Key Moments

"""
        
        # Add representative messages
        sample_messages = chapter_msgs.sample(min(5, len(chapter_msgs)))
        
        for _, msg in sample_messages.iterrows():
            sender = msg['sender']
            message = msg['message'][:100] + '...' if len(str(msg['message'])) > 100 else msg['message']
            narrative += f"- **{sender}**: {message}\n"
        
        return narrative
    
    def generate_prologue(self) -> str:
        """Generate overview of entire chat relationship"""
        total_messages = len(self.df)
        participants = self.df['sender'].unique().tolist()
        date_range = f"{self.df['date'].min()} to {self.df['date'].max()}"
        
        # Calculate relationship intensity
        days_span = (pd.to_datetime(self.df['date'].max()) - pd.to_datetime(self.df['date'].min())).days
        avg_daily = total_messages / max(days_span, 1)
        
        prologue = f"""
# Chat Story: Relationship Overview

## The Beginning

This conversation between **{', '.join(participants)}** spans from **{date_range}**, 
covering approximately **{days_span} days** with **{total_messages:,} total messages**.

### Relationship Dynamics

- **Average daily messages:** {avg_daily:.1f}
- **Most active participant:** {self.df['sender'].mode().iloc[0]}
- **Conversation intensity:** {'High' if avg_daily > 50 else 'Moderate' if avg_daily > 10 else 'Casual'}

### Communication Style

"""
        
        # Add media stats
        if 'media_type' in self.df.columns:
            media_dist = self.df['media_type'].value_counts()
            prologue += f"- **Text messages:** {media_dist.get('text', 0):,}\n"
            prologue += f"- **Links shared:** {media_dist.get('link', 0):,}\n"
            prologue += f"- **Media omitted:** {media_dist.get('media_omitted', 0):,}\n"
        
        return prologue
    
    def generate_epilogue(self) -> str:
        """Generate closing summary of relationship evolution"""
        # Split into first and second half
        self.df = self.df.sort_values('date')
        mid_point = len(self.df) // 2
        
        first_half = self.df.iloc[:mid_point]
        second_half = self.df.iloc[mid_point:]
        
        # Compare activity
        first_avg = first_half.groupby('sender').size().mean()
        second_avg = second_half.groupby('sender').size().mean()
        
        epilogue = f"""
# Epilogue: How Things Evolved

## Activity Changes

- **First half average:** {first_avg:.1f} messages per participant
- **Second half average:** {second_avg:.1f} messages per participant
- **Trend:** {'Increasing engagement' if second_avg > first_avg * 1.1 else 'Decreasing engagement' if second_avg < first_avg * 0.9 else 'Stable engagement'}

## Relationship Evolution

"""
        
        # Add sentiment evolution if available
        if 'compound' in self.df.columns:
            first_sentiment = first_half['compound'].mean()
            second_sentiment = second_half['compound'].mean()
            
            epilogue += f"- **Early sentiment:** {first_sentiment:.2f} ({'positive' if first_sentiment > 0.05 else 'negative' if first_sentiment < -0.05 else 'neutral'})\n"
            epilogue += f"- **Later sentiment:** {second_sentiment:.2f} ({'positive' if second_sentiment > 0.05 else 'negative' if second_sentiment < -0.05 else 'neutral'})\n"
        
        return epilogue
    
    def export_book_format(self, output_path: str = 'chat_story.md'):
        """Export entire chat as formatted book"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_prologue())
            f.write("\n\n---\n\n")
            
            if self.chapter_df is not None:
                for chapter_id in self.chapter_df['chapter_id'].unique():
                    f.write(self.generate_chapter_narrative(chapter_id))
                    f.write("\n\n---\n\n")
            
            f.write(self.generate_epilogue())
        
        return output_path
