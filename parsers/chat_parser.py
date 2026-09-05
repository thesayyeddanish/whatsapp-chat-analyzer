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
        try:
            # Try whatstk first
            self.df = whatstk.df_from_whatsapp(self.file_path)
        except Exception as e:
            # Fallback: manual parsing
            print(f"whatstk failed: {str(e)}. Trying manual parsing...")
            self.df = self._manual_parse()
        
        # Check if we have the right columns
        if self.df is None or len(self.df) == 0:
            raise ValueError("Failed to parse chat file. The file appears to be empty or invalid.")
        
        # Normalize column names
        self.df = self._normalize_columns()
        
        self.participants = self.df['sender'].unique().tolist()
        
        # Add metadata columns
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['message_length'] = self.df['message'].str.len()
        self.df['word_count'] = self.df['message'].str.split().str.len()
        
        # Detect media types
        self.df['media_type'] = self.df['message'].apply(self._detect_media)
        
        return self.df
    
    def _normalize_columns(self) -> pd.DataFrame:
        """Normalize column names from different WhatsApp formats"""
        # Map common column name variations
        column_mapping = {
            'sender': ['sender', 'from', 'user', 'name', 'author'],
            'message': ['message', 'text', 'msg', 'content'],
            'date': ['date', 'datetime', 'timestamp', 'time']
        }
        
        # Create reverse mapping
        reverse_mapping = {}
        for standard_name, variations in column_mapping.items():
            for var in variations:
                reverse_mapping[var] = standard_name
        
        # Rename columns
        new_columns = {}
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in reverse_mapping:
                new_columns[col] = reverse_mapping[col_lower]
            else:
                new_columns[col] = col
        
        self.df = self.df.rename(columns=new_columns)
        
        # Check required columns exist
        required_cols = ['sender', 'message', 'date']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(self.df.columns)}")
        
        return self.df
    
    def _manual_parse(self) -> pd.DataFrame:
        """Manual parsing for unsupported WhatsApp formats"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        messages = []
        current_sender = None
        current_message = []
        current_date = None
        
        # Common WhatsApp date patterns
        date_patterns = [
            r'^\[(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s+(.*?):\s*(.*)',  # [date] sender: message
            r'^(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?),\s+(.*?):\s*(.*)',  # date, sender: message
            r'^(\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?)\s+(.*?):\s*(.*)',  # date-time sender: message
            r'^(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2})\s+(.*?):\s*(.*)',  # date time sender: message
            r'^(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})\s+(.*?):\s*(.*)',  # European format
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match date patterns
            matched = False
            for pattern in date_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous message
                    if current_sender and current_message:
                        messages.append({
                            'date': current_date,
                            'sender': current_sender,
                            'message': ' '.join(current_message)
                        })
                    
                    # Start new message
                    current_date = match.group(1)
                    current_sender = match.group(2)
                    current_message = [match.group(3)]
                    matched = True
                    break
            
            if not matched:
                # Continuation of previous message
                if current_sender:
                    current_message.append(line)
        
        # Don't forget the last message
        if current_sender and current_message:
            messages.append({
                'date': current_date,
                'sender': current_sender,
                'message': ' '.join(current_message)
            })
        
        if len(messages) == 0:
            return None
        
        df = pd.DataFrame(messages)
        
        # Try to parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
        
        # If date parsing failed, try with dayfirst=True
        if df['date'].isna().all():
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        
        return df
    
    def _detect_media(self, message: str) -> str:
        """Classify message as text, image, video, link, voice, sticker, etc."""
        if message is None:
            return 'text'
        
        message_lower = message.lower()
        
        if '<Media omitted>' in message or 'omitted' in message_lower:
            return 'media_omitted'
        elif re.search(r'https?://', message):
            return 'link'
        elif message.startswith('📷') or 'photo' in message_lower:
            return 'image'
        elif message.startswith('🎥') or 'video' in message_lower:
            return 'video'
        elif message.startswith('🎤') or 'voice' in message_lower:
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
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            },
            'media_breakdown': self.df['media_type'].value_counts().to_dict()
        }
