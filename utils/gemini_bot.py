import pandas as pd
import google.generativeai as genai
from typing import Dict, Tuple

class GeminiChatAnalyzer:
    """AI-powered chat analyzer using Google Gemini"""
    
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

        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create context about the chat
        self.chat_context = self._create_chat_context()
    
    def _create_chat_context(self) -> str:
        """Create a summary of the chat for context"""
        context = f"""
This is a WhatsApp chat analysis with the following stats:
- Total Messages: {self.metadata['total_messages']:,}
- Total Words: {self.metadata['total_words']:,}
- Participants: {', '.join(self.metadata['participants'])}
- Date Range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}

Sample messages from the chat:
"""
        # Safely extract sample messages (handling small datasets without duplicate rows)
        if len(self.df) <= 20:
            samples = self.df
        else:
            samples = pd.concat([self.df.head(10), self.df.tail(10)]).drop_duplicates()

        for _, msg in samples.iterrows():
            sender = msg.get('sender', 'Unknown')
            date_str = msg.get('date', '')
            text = str(msg.get('message', ''))[:100]
            context += f"[{date_str}] {sender}: {text}\n"
        
        return context
    
    def analyze_chat(self, question: str) -> str:
        """Ask Gemini to analyze the chat"""
        # Calculate active span in days safely
        try:
            start_dt = pd.to_datetime(self.metadata['date_range']['start'])
            end_dt = pd.to_datetime(self.metadata['date_range']['end'])
            days_span = max((end_dt - start_dt).days + 1, 1)
        except Exception:
            days_span = 1

        most_active_hour = self.df.groupby('hour').size().idxmax() if 'hour' in self.df.columns and not self.df['hour'].isnull().all() else 'N/A'
        most_active_day = self.df.groupby('day_of_week').size().idxmax() if 'day_of_week' in self.df.columns and not self.df['day_of_week'].isnull().all() else 'N/A'
        
        prompt = f"""
You are an expert chat analyst. I'll give you information about a WhatsApp chat and you need to answer questions about it.

{self.chat_context}

Here are more detailed statistics:
- Message count by participant: {self.df.groupby('sender').size().to_dict()}
- Most active hour: {most_active_hour}:00
- Most active day: {most_active_day}
- Average messages per day: {len(self.df) / days_span:.1f}

The user's question is: "{question}"

Provide a detailed, friendly, and insightful answer. Use emojis to make it engaging. If the question is about finding specific messages or patterns, analyze the data carefully and provide accurate information.

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def search_messages(self, query: str) -> Tuple[str, pd.DataFrame]:
        """Search for messages and get AI analysis"""
        
        mask = self.df['message'].astype(str).str.lower().str.contains(query.lower(), case=False, regex=False, na=False)
        results = self.df[mask].copy()
        
        if len(results) == 0:
            return f"🔍 No messages found containing '{query}'", pd.DataFrame()
        
        prompt = f"""
The user searched for: "{query}"

I found {len(results)} matching messages in the chat.

Here are some sample messages:
"""
        for _, msg in results.head(10).iterrows():
            prompt += f"[{msg.get('date', '')}] {msg.get('sender', '')}: {str(msg.get('message', ''))[:150]}\n"
        
        prompt += f"""

Based on these search results, provide:
1. A brief summary of what these messages are about
2. Who sent most of these messages
3. Any interesting patterns or insights
4. The context or time period when these messages were sent

Make it conversational and friendly with emojis!"""
        
        try:
            response = self.model.generate_content(prompt)
            analysis = response.text
            full_response = f"🔍 **Found {len(results):,} messages** containing '{query}'\n\n{analysis}"
            return full_response, results.head(10)
        except Exception as e:
            return f"Found {len(results)} messages, but couldn't analyze them: {str(e)}", results.head(10)
    
    def get_insights(self) -> str:
        """Get AI-generated insights about the chat"""
        try:
            start_dt = pd.to_datetime(self.metadata['date_range']['start'])
            end_dt = pd.to_datetime(self.metadata['date_range']['end'])
            days_active = max((end_dt - start_dt).days + 1, 1)
        except Exception:
            days_active = 1

        # Safe sampling
        sample_size = min(30, len(self.df))
        sampled_df = self.df.sample(sample_size) if sample_size > 0 else self.df

        prompt = f"""
You are a relationship and communication expert. Analyze this WhatsApp chat and provide deep insights.

Chat Statistics:
- Total Messages: {self.metadata['total_messages']:,}
- Total Words: {self.metadata['total_words']:,}
- Participants: {', '.join(self.metadata['participants'])}
- Date Range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}
- Days Active: {days_active:,}

Message distribution: {self.df.groupby('sender').size().to_dict()}

Sample messages:
"""
        for _, msg in sampled_df.iterrows():
            prompt += f"[{msg.get('sender', '')}]: {str(msg.get('message', ''))[:100]}\n"
        
        prompt += """

Provide a comprehensive analysis including:
1. **Communication Style** - How do these people communicate? (formal, casual, friendly, etc.)
2. **Relationship Dynamics** - What does this chat reveal about their relationship?
3. **Conversation Patterns** - Any interesting patterns in how they talk?
4. **Emotional Tone** - What's the overall emotional vibe?
5. **Key Topics** - What do they talk about most?
6. **Fun Observations** - Any quirky or interesting things you notice

Make it engaging, insightful, and friendly with emojis. Write it like you're telling a story about their friendship/relationship."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Couldn't generate insights: {str(e)}"
    
    def summarize_week(self, week_start: str, week_end: str) -> str:
        """Summarize a specific week of the chat"""
        
        week_mask = (self.df['date'] >= week_start) & (self.df['date'] <= week_end)
        week_messages = self.df[week_mask]
        
        if len(week_messages) == 0:
            return "No messages found for that week."
        
        sample_size = min(50, len(week_messages))
        sampled_week = week_messages.sample(sample_size)

        prompt = f"""
Summarize this week of WhatsApp chat (from {week_start} to {week_end}).

Total messages this week: {len(week_messages)}

Sample messages:
"""
        for _, msg in sampled_week.iterrows():
            prompt += f"[{msg.get('sender', '')}]: {str(msg.get('message', ''))[:100]}\n"
        
        prompt += """

Provide a weekly summary like a story:
1. What were the main topics discussed?
2. Any important events or decisions?
3. How was the mood/tone this week?
4. Who was most active?
5. Any memorable moments?

Make it read like a weekly newsletter or diary entry. Fun and engaging with emojis!"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Couldn't summarize week: {str(e)}"
    
    def compare_participants(self) -> str:
        """Compare communication styles of participants"""
        
        prompt = f"""
Compare the communication styles of these WhatsApp chat participants.

Chat Statistics:
- Participants: {', '.join(self.metadata['participants'])}
"""
        
        for sender in self.metadata['participants']:
            sender_msgs = self.df[self.df['sender'] == sender]
            if len(sender_msgs) == 0:
                continue
            
            avg_len = sender_msgs['message_length'].mean() if 'message_length' in sender_msgs.columns else 0.0
            sample_size = min(5, len(sender_msgs))
            sampled_sender = sender_msgs.sample(sample_size)

            prompt += f"""
**{sender}**:
- Messages: {len(sender_msgs)}
- Average message length: {avg_len:.1f} characters
- Sample messages:
"""
            for _, msg in sampled_sender.iterrows():
                prompt += f"  - {str(msg.get('message', ''))[:100]}\n"
        
        prompt += """

Compare and contrast:
1. **Communication Style** - Who's more verbose? Who's concise?
2. **Tone** - Who's more positive/emotional/factual?
3. **Topics** - Does each person have preferred topics?
4. **Response Patterns** - Who initiates more? Who replies faster?
5. **Personality Insights** - What can you infer about each person's personality?

Make it fun, insightful, and friendly with emojis!"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Couldn't compare participants: {str(e)}"
