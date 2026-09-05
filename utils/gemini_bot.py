import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Optional
import os

class GeminiChatAnalyzer:
    """AI-powered chat analyzer using Google Gemini"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict, api_key: str):
        self.df = df.copy()
        self.metadata = metadata
        self.api_key = api_key
        
        # Initialize Gemini
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
        # Add sample messages (first 20 and last 20)
        samples = pd.concat([self.df.head(10), self.df.tail(10)])
        for _, msg in samples.iterrows():
            context += f"[{msg['date']}] {msg['sender']}: {msg['message'][:100]}\n"
        
        return context
    
    def analyze_chat(self, question: str) -> str:
        """Ask Gemini to analyze the chat"""
        
        prompt = f"""
You are an expert chat analyst. I'll give you information about a WhatsApp chat and you need to answer questions about it.

{self.chat_context}

Here are more detailed statistics:
- Message count by participant: {self.df.groupby('sender').size().to_dict()}
- Most active hour: {self.df.groupby('hour').size().idxmax()}:00
- Most active day: {self.df.groupby('day_of_week').size().idxmax()}
- Average messages per day: {len(self.df) / max((pd.to_datetime(self.metadata['date_range']['end']) - pd.to_datetime(self.metadata['date_range']['start'])).days + 1, 1):.1f}

The user's question is: "{question}"

Provide a detailed, friendly, and insightful answer. Use emojis to make it engaging. If the question is about finding specific messages or patterns, analyze the data carefully and provide accurate information.

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def search_messages(self, query: str) -> tuple:
        """Search for messages and get AI analysis"""
        
        # First, find matching messages
        mask = self.df['message'].str.lower().str.contains(query.lower(), case=False, regex=False, na=False)
        results = self.df[mask].copy()
        
        if len(results) == 0:
            return f"🔍 No messages found containing '{query}'", pd.DataFrame()
        
        # Get AI analysis of the search results
        prompt = f"""
The user searched for: "{query}"

I found {len(results)} matching messages in the chat.

Here are some sample messages:
"""
        for _, msg in results.head(10).iterrows():
            prompt += f"[{msg['date']}] {msg['sender']}: {msg['message'][:150]}\n"
        
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
        
        prompt = f"""
You are a relationship and communication expert. Analyze this WhatsApp chat and provide deep insights.

Chat Statistics:
- Total Messages: {self.metadata['total_messages']:,}
- Total Words: {self.metadata['total_words']:,}
- Participants: {', '.join(self.metadata['participants'])}
- Date Range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}
- Days Active: {(pd.to_datetime(self.metadata['date_range']['end']) - pd.to_datetime(self.metadata['date_range']['start'])).days + 1:,}

Message distribution: {self.df.groupby('sender').size().to_dict()}

Sample messages:
"""
        for _, msg in self.df.sample(30).iterrows():
            prompt += f"[{msg['sender']}]: {msg['message'][:100]}\n"
        
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
        
        # Filter messages for that week
        week_mask = (self.df['date'] >= week_start) & (self.df['date'] <= week_end)
        week_messages = self.df[week_mask]
        
        if len(week_messages) == 0:
            return "No messages found for that week."
        
        prompt = f"""
Summarize this week of WhatsApp chat (from {week_start} to {week_end}).

Total messages this week: {len(week_messages)}

Sample messages:
"""
        for _, msg in week_messages.sample(min(50, len(week_messages))).iterrows():
            prompt += f"[{msg['sender']}]: {msg['message'][:100]}\n"
        
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
            prompt += f"""
**{sender}**:
- Messages: {len(sender_msgs)}
- Average message length: {sender_msgs['message_length'].mean():.1f} characters
- Sample messages:
"""
            for _, msg in sender_msgs.sample(5).iterrows():
                prompt += f"  - {msg['message'][:100]}\n"
        
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
