import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Tuple
import whatstk

class WhatsAppChatParser:
    """Parse WhatsApp .txt exports into structured DataFrame"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.participants = []
        
    def parse(self) -> pd.DataFrame:
        """Load chat using whatstk with automatic format detection"""
        self.df = whatstk.df_from_whatsapp(self.file_path)
        self.participants = self.df['sender'].unique().tolist()
        
        # Add metadata columns
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        self.df['hour'] = pd.to_datetime(self.df['date']).dt.hour
        self.df['day_of_week'] = pd.to_datetime(self.df['date']).dt.day_name()
        self.df['message_length'] = self.df['message'].str.len()
        self.df['word_count'] = self.df['message'].str.split().str.len()
        
        # Detect media types
        self.df['media_type'] = self.df['message'].apply(self._detect_media)
        
        return self.df
    
    def _detect_media(self, message: str) -> str:
        """Classify message as text, image, video, link, voice, sticker, etc."""
        if message.startswith('<Media omitted>') or 'omitted' in message.lower():
            return 'media_omitted'
        elif re.search(r'https?://', message):
            return 'link'
        elif message.startswith('📷') or 'photo' in message.lower():
            return 'image'
        elif message.startswith('🎥') or 'video' in message.lower():
            return 'video'
        elif message.startswith('🎤') or 'voice' in message.lower():
            return 'voice_note'
        elif message.startswith('📍'):
            return 'location'
        elif len(message) <= 5 and any(c in message for c in '😀😂❤️👍🔥'):
            return 'emoji_only'
        else:
            return 'text'
    
    def get_chat_metadata(self) -> Dict:
        """Return basic chat statistics"""
        return {
            'total_messages': len(self.df),
            'total_words': self.df['word_count'].sum(),
            'total_characters': self.df['message'].str.len().sum(),
            'participants': self.participants,
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            },
            'media_breakdown': self.df['media_type'].value_counts().to_dict()
        }
