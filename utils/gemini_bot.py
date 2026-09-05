import pandas as pd
import google.generativeai as genai
import re
from typing import Dict, Tuple

class GeminiChatAnalyzer:
    """AI-powered chat analyzer reading full multi-year WhatsApp logs"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict, api_key: str):
        self.df = df.copy()
        self.metadata = metadata
        self.api_key = api_key
        
        # Ensure datetime and string types
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
        if 'hour' not in self.df.columns and 'date' in self.df.columns:
            self.df['hour'] = self.df['date'].dt.hour
            
        if 'day_of_week' not in self.df.columns and 'date' in self.df.columns:
            self.df['day_of_week'] = self.df['date'].dt.day_name()
            
        if 'message_length' not in self.df.columns and 'message' in self.df.columns:
            self.df['message_length'] = self.df['message'].astype(str).str.len()

        # Configure API
        genai.configure(api_key=api_key)

    def _generate_response(self, prompt: str) -> str:
        """Safe generation wrapper using available Gemini models"""
        models_to_try = [
            'gemini-3.6-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro'
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                last_error = e
                continue
                
        return f"Sorry, I encountered an error across available models: {str(last_error)}"

    def _build_chat_text_from_df(self, dataframe: pd.DataFrame, max_chars: int = 2_500_000) -> str:
        """Formats a DataFrame into readable conversation lines for Gemini"""
        lines = []
        for _, row in dataframe.iterrows():
            sender = row.get('sender', 'Unknown')
            date_str = str(row.get('date', ''))
            msg = str(row.get('message', '')).replace('\n', ' ')
            lines.append(f"[{date_str}] {sender}: {msg}")
        
        full_text = "\n".join(lines)
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...[Truncated due to size limit]..."
        return full_text

    def analyze_chat(self, question: str) -> str:
        """Dynamically fetches relevant logs across all years to answer user questions"""
        
        # Check if user is asking about specific dates/years/keywords
        q_lower = question.lower()
        filtered_df = self.df
        
        # Check for specific year mentioned in query (e.g., 2023, 2022)
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            target_year = int(year_match.group(1))
            year_mask = self.df['date'].dt.year == target_year
            if year_mask.any():
                filtered_df = self.df[year_mask]

        # Convert the target dataset (or full chat if no specific filter) to text
        chat_logs = self._build_chat_text_from_df(filtered_df)
        
        prompt = f"""
You are an expert chat analyst. You are provided with the actual message logs from a WhatsApp chat archive.

Chat Overview:
- Total Messages in Archive: {self.metadata['total_messages']:,}
- Total Messages in Context Below: {len(filtered_df):,}
- Participants: {', '.join(self.metadata['participants'])}
- Date Range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}

Actual Chat Logs:
{chat_logs}

User Question: "{question}"

Instructions:
- Answer the user's question directly by reading through the provided chat logs above.
- Do NOT claim that messages are missing if they exist in the provided chat logs.
- Provide a clear, detailed, and friendly response with emojis!
"""
        return self._generate_response(prompt)
    
    def search_messages(self, query: str) -> Tuple[str, pd.DataFrame]:
        """Search for messages across the full dataset"""
        mask = self.df['message'].astype(str).str.lower().str.contains(query.lower(), case=False, regex=False, na=False)
        results = self.df[mask].copy()
        
        if len(results) == 0:
            return f"🔍 No messages found containing '{query}'", pd.DataFrame()
        
        matches_text = self._build_chat_text_from_df(results.head(500))
        
        prompt = f"""
The user searched for: "{query}"

Found {len(results)} matching messages across the full chat history.

Matching Message Logs:
{matches_text}

Based on all matching messages across the full timeline, provide:
1. A summary of what these discussions were about over time
2. Who mentioned this topic most frequently
3. How discussions on this topic evolved over time

Make it conversational and clear with emojis!"""
        
        analysis = self._generate_response(prompt)
        full_response = f"🔍 **Found {len(results):,} messages** containing '{query}'\n\n{analysis}"
        return full_response, results.head(20)
    
    def get_insights(self) -> str:
        """Get AI-generated insights across the full conversation history"""
        chat_logs = self._build_chat_text_from_df(self.df)

        prompt = f"""
You are a relationship and communication expert. You are reviewing the full WhatsApp chat archive provided below.

Complete Chat Logs:
{chat_logs}

Provide a comprehensive synthesis report including:
1. **Evolution Over Time** - How did communication change across the years?
2. **Relationship Dynamics & Growth** - What does the timeline reveal about their bond?
3. **Key Themes & Eras** - Main topics discussed during different periods.
4. **Communication Style & Tone** - How participants express affection, humor, or conflict.
5. **Memorable Patterns & Quirks** - Unique habits or recurring jokes across the archive.

Make it engaging, empathetic, and organized with clear section headers and emojis."""
        
        return self._generate_response(prompt)
    
    def summarize_week(self, week_start: str, week_end: str) -> str:
        """Summarize a specific week using exact messages from that date window"""
        week_mask = (self.df['date'] >= week_start) & (self.df['date'] <= week_end)
        week_messages = self.df[week_mask]
        
        if len(week_messages) == 0:
            return f"No messages found for the period {week_start} to {week_end}."
        
        week_text = self._build_chat_text_from_df(week_messages)

        prompt = f"""
Summarize this specific week of chat ({week_start} to {week_end}).

Total messages in this period: {len(week_messages)}

Full Messages for this Week:
{week_text}

Provide a detailed summary:
1. Primary topics and discussions
2. Key events, decisions, or plans made
3. Emotional mood and interactions between participants

Make it read like a clear journal entry with emojis!"""
        
        return self._generate_response(prompt)
    
    def compare_participants(self) -> str:
        """Compare communication styles using full chat logs"""
        chat_logs = self._build_chat_text_from_df(self.df, max_chars=1_500_000)
        
        prompt = f"""
Compare the communication styles of these WhatsApp chat participants using their full chat history.

Participants: {', '.join(self.metadata['participants'])}
Total Messages: {self.metadata['total_messages']:}

Chat Logs Context:
{chat_logs}

Compare and contrast:
1. **Verbosity & Tone** - Who sends longer messages? Who initiates conversations more often?
2. **Emotional Expressiveness** - How does each participant share feelings or humor over time?
3. **Topic Preferences** - What distinct subjects does each person bring up most frequently?
4. **Personality Summary** - What does the data reveal about each person's chat personality?

Make it insightful, balanced, and fun with emojis!"""
        
        return self._generate_response(prompt)
