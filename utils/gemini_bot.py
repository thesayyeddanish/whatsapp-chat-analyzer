import pandas as pd
import google.generativeai as genai
from typing import Dict, Tuple

class GeminiChatAnalyzer:
    """AI-powered chat analyzer using Google Gemini across full multi-year chat logs"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict, api_key: str):
        self.df = df.copy()
        self.metadata = metadata
        self.api_key = api_key
        
        # Ensure requisite datetime helper columns exist
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
        if 'hour' not in self.df.columns and 'date' in self.df.columns:
            self.df['hour'] = self.df['date'].dt.hour
            
        if 'day_of_week' not in self.df.columns and 'date' in self.df.columns:
            self.df['day_of_week'] = self.df['date'].dt.day_name()
            
        if 'message_length' not in self.df.columns and 'message' in self.df.columns:
            self.df['message_length'] = self.df['message'].astype(str).str.len()

        # Configure API without blocking UI initialization
        genai.configure(api_key=api_key)

    def _generate_response(self, prompt: str) -> str:
        """Safe generation wrapper using Gemini's large-context models"""
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

    def _build_full_chat_text(self, max_chars: int = 3_000_000) -> str:
        """Formats the ENTIRE dataset into a single text block for Gemini's context window"""
        lines = []
        for _, row in self.df.iterrows():
            sender = row.get('sender', 'Unknown')
            date_str = str(row.get('date', ''))
            msg = str(row.get('message', '')).replace('\n', ' ')
            lines.append(f"[{date_str}] {sender}: {msg}")
        
        full_text = "\n".join(lines)
        
        # Safe character truncation to protect against extreme payload limits
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...[Chat log truncated due to extreme size]..."
        return full_text
    
    def analyze_chat(self, question: str) -> str:
        """Ask Gemini to analyze the complete chat log"""
        full_chat = self._build_full_chat_text()
        
        prompt = f"""
You are an expert chat analyst. Below is the COMPLETE chat log spanning the entire history of this conversation.

Chat Overview:
- Total Messages: {self.metadata['total_messages']:,}
- Total Words: {self.metadata['total_words']:,}
- Participants: {', '.join(self.metadata['participants'])}
- Date Range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}

Entire Chat History:
{full_chat}

User Question: "{question}"

Instructions:
- Carefully analyze the full timeline to answer the user's question accurately.
- Reference specific years, dates, or milestones if relevant.
- Provide a clear, detailed, and friendly response with emojis!
"""
        return self._generate_response(prompt)
    
    def search_messages(self, query: str) -> Tuple[str, pd.DataFrame]:
        """Search for messages across the entire dataset and analyze matches"""
        mask = self.df['message'].astype(str).str.lower().str.contains(query.lower(), case=False, regex=False, na=False)
        results = self.df[mask].copy()
        
        if len(results) == 0:
            return f"🔍 No messages found containing '{query}'", pd.DataFrame()
        
        matching_lines = []
        for _, msg in results.iterrows():
            matching_lines.append(f"[{msg.get('date', '')}] {msg.get('sender', '')}: {str(msg.get('message', ''))}")
        
        matches_text = "\n".join(matching_lines[:500]) # Pass up to 500 matching results directly
        
        prompt = f"""
The user searched for: "{query}"

Found {len(results)} matching messages across the full chat history.

Matching Messages Log:
{matches_text}

Based on all matching messages across the full timeline, provide:
1. A summary of what these discussions were about over time
2. Who mentioned this topic most frequently
3. How discussions on this topic evolved from early dates to recent dates

Make it conversational and clear with emojis!"""
        
        analysis = self._generate_response(prompt)
        full_response = f"🔍 **Found {len(results):,} messages** containing '{query}'\n\n{analysis}"
        return full_response, results.head(20)
    
    def get_insights(self) -> str:
        """Get AI-generated insights across all 4 years of conversation history"""
        full_chat = self._build_full_chat_text()

        prompt = f"""
You are a relationship and communication expert. You are reviewing the ENTIRE 4-year WhatsApp chat archive provided below.

Complete Chat Log:
{full_chat}

Provide a comprehensive 4-year synthesis report including:
1. **Evolution Over Time** - How did communication change from Year 1 to Year 4?
2. **Relationship Dynamics & Growth** - What does the full timeline reveal about their bond and changing closeness?
3. **Key Themes & Eras** - What were the distinct phases or main topics during different periods?
4. **Communication Style & Tone** - How do participants express affection, humor, or conflict over time?
5. **Memorable Patterns & Quirks** - Unique habits or recurring jokes noticed across the entire archive.

Make it engaging, empathetic, and organized with clear section headers and emojis."""
        
        return self._generate_response(prompt)
    
    def summarize_week(self, week_start: str, week_end: str) -> str:
        """Summarize a specific week of the chat using all messages from that window"""
        week_mask = (self.df['date'] >= week_start) & (self.df['date'] <= week_end)
        week_messages = self.df[week_mask]
        
        if len(week_messages) == 0:
            return "No messages found for that specified timeframe."
        
        week_text = "\n".join(
            f"[{msg.get('date', '')}] {msg.get('sender', '')}: {msg.get('message', '')}"
            for _, msg in week_messages.iterrows()
        )

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
        """Compare communication styles using metrics and full chat context"""
        full_chat = self._build_full_chat_text(max_chars=1_500_000)
        
        prompt = f"""
Compare the communication styles of these WhatsApp chat participants using their FULL history.

Participants: {', '.join(self.metadata['participants'])}
Total Messages: {self.metadata['total_messages']:}

Chat History Context:
{full_chat}

Compare and contrast:
1. **Verbosity & Tone** - Who sends longer messages? Who initiates conversations more often?
2. **Emotional Expressiveness** - How does each participant share feelings or humor over time?
3. **Topic Preferences** - What distinct subjects does each person bring up most frequently?
4. **Personality Summary** - What does 4 years of data reveal about each person's chat personality?

Make it insightful, balanced, and fun with emojis!"""
        
        return self._generate_response(prompt)
