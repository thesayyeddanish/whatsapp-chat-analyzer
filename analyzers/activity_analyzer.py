import pandas as pd
import numpy as np
from typing import Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go

class ActivityAnalyzer:
    """Analyze temporal patterns in chat activity"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_volume_patterns(self, freq: str = 'D') -> pd.DataFrame:
        """Aggregate messages by time frequency (D=day, W=week, M=month, Y=year)"""
        self.df['datetime'] = pd.to_datetime(self.df['date'])
        return self.df.groupby([pd.Grouper(key='datetime', freq=freq), 'sender']).agg({
            'message': 'count',
            'word_count': 'sum',
            'message_length': 'sum'
        }).reset_index()
    
    def get_active_hours(self) -> pd.DataFrame:
        """Find busiest hours of day"""
        hourly = self.df.groupby('hour').agg({
            'message': 'count',
            'sender': 'count'
        }).reset_index()
        hourly.columns = ['hour', 'message_count', 'total_activity']
        peak_hour = hourly.loc[hourly['message_count'].idxmax()]
        return hourly, peak_hour
    
    def get_active_days(self) -> pd.DataFrame:
        """Find most active days of week"""
        daily = self.df.groupby('day_of_week').size().reset_index(name='count')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily['day_of_week'] = pd.Categorical(daily['day_of_week'], categories=day_order, ordered=True)
        return daily.sort_values('count', ascending=False)
    
    def calculate_response_times(self) -> pd.DataFrame:
        """Calculate average response time per participant"""
        self.df = self.df.sort_values('date')
        response_times = []
        
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender].copy()
            sender_msgs['prev_time'] = sender_msgs['date'].shift(1)
            sender_msgs['response_time'] = (
                sender_msgs['date'] - sender_msgs['prev_time']
            ).dt.total_seconds() / 60  # in minutes
            
            avg_response = sender_msgs['response_time'].median()
            response_times.append({
                'sender': sender,
                'avg_response_time_min': avg_response,
                'total_replies': len(sender_msgs) - 1
            })
        
        return pd.DataFrame(response_times)
    
    def analyze_initiators(self, silence_threshold_hours: int = 6) -> pd.DataFrame:
        """Find who starts conversations after silence periods"""
        self.df = self.df.sort_values('date')
        self.df['time_diff'] = self.df['date'].diff().dt.total_seconds() / 3600
        
        # Messages after silence threshold
        conversation_starts = self.df[self.df['time_diff'] > silence_threshold_hours].copy()
        
        initiators = conversation_starts.groupby('sender').size().reset_index(name='conversations_started')
        initiators['percentage'] = (
            initiators['conversations_started'] / initiators['conversations_started'].sum() * 100
        )
        
        return initiators.sort_values('conversations_started', ascending=False)
    
    def create_heatmap(self, freq: str = 'hourly') -> go.Figure:
        """Create activity heatmap (hourly or daily)"""
        if freq == 'hourly':
            pivot = self.df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot['day_of_week'] = pd.Categorical(pivot['day_of_week'], categories=day_order, ordered=True)
            pivot = pivot.pivot(index='day_of_week', columns='hour', values='count')
            
            fig = px.imshow(
                pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                color_continuous_scale='YlOrRd',
                title='Chat Activity Heatmap by Hour & Day'
            )
            return fig
