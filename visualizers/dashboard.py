import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List
import streamlit as st

class ChatVisualizer:
    """Create interactive visualizations and infographics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_word_cloud(self, text: str, title: str = 'Word Cloud') -> go.Figure:
        """Generate word cloud visualization"""
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Convert to plotly
        wc_image = wordcloud.to_array()
        
        fig = px.imshow(
            wc_image,
            title=title,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        
        return fig
    
    def create_message_timeline(self) -> go.Figure:
        """Timeline of messages over time"""
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        
        daily_counts = self.df.groupby(
            [pd.Grouper(key='datetime', freq='D'), 'sender']
        ).size().reset_index(name='count')
        
        fig = px.area(
            daily_counts,
            x='datetime',
            y='count',
            color='sender',
            title='Message Volume Over Time',
            labels={'datetime': 'Date', 'count': 'Messages', 'sender': 'Participant'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Messages per Day',
            hovermode='x unified'
        )
        
        return fig
    
    def create_participant_comparison(self, leaderboard: pd.DataFrame) -> go.Figure:
        """Compare participants across metrics"""
        fig = go.Figure()
        
        # Add bars for different metrics
        fig.add_trace(go.Bar(
            name='Messages',
            x=leaderboard['sender'],
            y=leaderboard['message_count'],
            marker_color='#636EFA'
        ))
        
        fig.add_trace(go.Bar(
            name='Words',
            x=leaderboard['sender'],
            y=leaderboard['total_words'] / 100,  # Scale down
            marker_color='#EF553B'
        ))
        
        fig.update_layout(
            title='Participant Activity Comparison',
            barmode='group',
            xaxis_title='Participant',
            yaxis_title='Count (Words scaled by 100)',
            legend_title='Metric'
        )
        
        return fig
    
    def create_sentiment_timeline(self) -> go.Figure:
        """Sentiment over time"""
        if 'compound' not in self.df.columns:
            raise ValueError("Run sentiment analysis first")
        
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        
        daily_sentiment = self.df.groupby(
            pd.Grouper(key='datetime', freq='D')
        )['compound'].mean().reset_index()
        
        fig = px.line(
            daily_sentiment,
            x='datetime',
            y='compound',
            title='Average Sentiment Over Time',
            labels={'datetime': 'Date', 'compound': 'Sentiment Score'}
        )
        
        # Add threshold lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="Positive")
        fig.add_hline(y=-0.05, line_dash="dash", line_color="red", annotation_text="Negative")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig.update_layout(yaxis_range=[-1, 1])
        
        return fig
    
    def create_emoji_matrix(self, emoji_stats: Dict) -> go.Figure:
        """Emoji usage heatmap by participant"""
        # Convert emoji stats to DataFrame
        emoji_data = []
        for sender, stats in emoji_stats.items():
            for emoji_char, count in stats['top_emojis'][:10]:
                emoji_data.append({
                    'sender': sender,
                    'emoji': emoji_char,
                    'count': count
                })
        
        emoji_df = pd.DataFrame(emoji_data)
        
        if len(emoji_df) == 0:
            return go.Figure().add_annotation(text="No emoji data available")
        
        fig = px.density_heatmap(
            emoji_df,
            x='emoji',
            y='sender',
            z='count',
            title='Emoji Usage by Participant',
            color_continuous_scale='YlOrRd'
        )
        
        fig.update_layout(
            xaxis_title='Emoji',
            yaxis_title='Participant'
        )
        
        return fig
    
    def create_infographic_card(self, stats: Dict, output_path: str = 'chat_stats.png'):
        """Generate shareable infographic card"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Chat Statistics Summary', fontsize=16, fontweight='bold')
        
        # Top left: Message distribution
        participants = list(stats['participants'])
        message_counts = [stats['message_count'].get(p, 0) for p in participants]
        
        axes[0, 0].bar(participants, message_counts, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0, 0].set_title('Messages per Participant')
        axes[0, 0].set_ylabel('Count')
        
        # Top right: Media breakdown
        if 'media_breakdown' in stats:
            media_types = list(stats['media_breakdown'].keys())[:5]
            media_counts = list(stats['media_breakdown'].values())[:5]
            axes[0, 1].pie(media_counts, labels=media_types, autopct='%1.1f%%')
            axes[0, 1].set_title('Media Type Distribution')
        
        # Bottom left: Activity by day
        if 'active_days' in stats:
            days = stats['active_days']['day_of_week'].tolist()[:7]
            counts = stats['active_days']['count'].tolist()[:7]
            axes[1, 0].bar(days, counts, color='#9b59b6')
            axes[1, 0].set_title('Messages by Day of Week')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Bottom right: Top stats
        summary_text = f"""
        Total Messages: {stats.get('total_messages', 0):,}
        Total Words: {stats.get('total_words', 0):,}
        Participants: {len(stats.get('participants', []))}
        Date Range: {stats.get('date_range', {}).get('start', 'N/A')} to {stats.get('date_range', {}).get('end', 'N/A')}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
