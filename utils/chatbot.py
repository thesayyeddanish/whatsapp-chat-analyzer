import pandas as pd
import re
from typing import Dict, List, Tuple
from sentiment.sentiment_analyzer import SentimentAnalyzer

class ChatAnalyzerBot:
    """Intelligent chatbot that answers questions about the chat"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df.copy()
        self.metadata = metadata
        self.participants = metadata['participants']
        
    def answer_question(self, question: str) -> str:
        """Process user question and return answer"""
        question_lower = question.lower().strip()
        
        # Remove punctuation
        question_clean = re.sub(r'[^\w\s]', ' ', question_lower)
        
        # Route to appropriate handler
        if any(word in question_clean for word in ['who talk', 'who send', 'who message', 'more message', 'most message']):
            return self._who_talks_more()
        
        elif any(word in question_clean for word in ['active hour', 'peak hour', 'busy hour', 'what time']):
            return self._peak_hour()
        
        elif any(word in question_clean for word in ['active day', 'peak day', 'busy day', 'what day']):
            return self._peak_day()
        
        elif any(word in question_clean for word in ['total message', 'how many message', 'message count']):
            return self._total_messages()
        
        elif any(word in question_clean for word in ['positive', 'happy', 'good mood']):
            return self._positive_messages()
        
        elif any(word in question_clean for word in ['negative', 'sad', 'bad mood', 'angry']):
            return self._negative_messages()
        
        elif any(word in question_clean for word in ['word', 'vocabulary', 'common word', 'frequent word']):
            # Check if asking about specific person
            for participant in self.participants:
                if participant.lower() in question_clean:
                    return self._top_words(participant)
            return self._top_words(None)
        
        elif any(word in question_clean for word in ['emoji', 'emoticon']):
            return self._emoji_usage()
        
        elif any(word in question_clean for word in ['link', 'url', 'website']):
            return self._link_sharing()
        
        elif any(word in question_clean for word in ['first message', 'oldest message', 'earliest message']):
            return self._first_message()
        
        elif any(word in question_clean for word in ['last message', 'recent message', 'newest message']):
            return self._last_message()
        
        elif any(word in question_clean for word in ['average', 'avg', 'mean']):
            if 'message' in question_clean or 'per day' in question_clean:
                return self._average_messages_per_day()
        
        elif any(word in question_clean for word in ['longest message', 'biggest message', 'largest message']):
            return self._longest_message()
        
        elif any(word in question_clean for word in ['shortest message', 'smallest message', 'smallest message']):
            return self._shortest_message()
        
        elif any(word in question_clean for word in ['who start', 'who initi', 'first to message']):
            return self._who_initiates()
        
        elif any(word in question_clean for word in ['response time', 'reply time', 'how fast']):
            return self._response_time()
        
        elif any(word in question_clean for word in ['help', 'what can', 'what question', 'example']):
            return self._help()
        
        else:
            return self._default_response()
    
    def _who_talks_more(self) -> str:
        """Answer: Who talks more?"""
        message_counts = self.df.groupby('sender').size()
        
        if len(message_counts) < 2:
            return "There's only one participant in this chat, so no comparison possible! 😊"
        
        top_talker = message_counts.idxmax()
        top_count = message_counts.max()
        second_talker = message_counts.idxmin() if len(message_counts) == 2 else message_counts.sort_values().iloc[-2]
        second_count = message_counts.min() if len(message_counts) == 2 else message_counts.sort_values().iloc[-2]
        
        ratio = top_count / second_count
        
        percentage = (top_count / len(self.df)) * 100
        
        response = f"📊 **{top_talker}** talks the most with **{top_count:,} messages** ({percentage:.1f}% of total).\n\n"
        
        if ratio > 2:
            response += f"That's {ratio:.1f}x more than {second_talker}! {top_talker} is definitely the main conversationalist here! 🏆"
        elif ratio > 1.5:
            response += f"About {ratio:.1f}x more than {second_talker}. Pretty active! 💬"
        else:
            response += f"Only {ratio:.1f}x more than {second_talker}. You both have a balanced conversation! 🤝"
        
        return response
    
    def _peak_hour(self) -> str:
        """Answer: What's the most active hour?"""
        hourly = self.df.groupby('hour').size()
        peak_hour = hourly.idxmax()
        peak_count = hourly.max()
        
        # Convert to 12-hour format
        if peak_hour == 0:
            time_str = "12:00 AM (midnight)"
        elif peak_hour < 12:
            time_str = f"{peak_hour}:00 AM"
        elif peak_hour == 12:
            time_str = "12:00 PM (noon)"
        else:
            time_str = f"{peak_hour - 12}:00 PM"
        
        response = f"🕐 Your **peak hour is {time_str}** with **{peak_count:,} messages**!\n\n"
        
        if 6 <= peak_hour <= 9:
            response += "Looks like you're early birds! 🌅"
        elif 12 <= peak_hour <= 14:
            response += "Lunch break chats? 🍕"
        elif 18 <= peak_hour <= 22:
            response += "Evening conversations are your favorite! 🌙"
        elif peak_hour >= 23 or peak_hour <= 2:
            response += "Night owls! 🦉🌃"
        else:
            response += "Interesting timing pattern! ⏰"
        
        return response
    
    def _peak_day(self) -> str:
        """Answer: What's the most active day?"""
        daily = self.df.groupby('day_of_week').size()
        peak_day = daily.idxmax()
        peak_count = daily.max()
        
        response = f"📅 **{peak_day}** is your most active day with **{peak_count:,} messages**!\n\n"
        
        if peak_day in ['Saturday', 'Sunday']:
            response += "Weekend vibes! You love chatting on weekends! 🎉"
        elif peak_day == 'Monday':
            response += "Starting the week strong with lots of messages! 💪"
        elif peak_day == 'Friday':
            response += "Friday feels! Getting ready for the weekend! 🎊"
        else:
            response += "Midweek is when you chat the most! 📱"
        
        return response
    
    def _total_messages(self) -> str:
        """Answer: Total messages"""
        total = len(self.df)
        words = self.df['word_count'].sum()
        days = (pd.to_datetime(self.metadata['date_range']['end']) - pd.to_datetime(self.metadata['date_range']['start'])).days + 1
        avg_per_day = total / max(days, 1)
        
        response = f"💬 **Total Messages:** {total:,}\n\n"
        response += f"📝 **Total Words:** {words:,}\n\n"
        response += f"📅 **Days Active:** {days:,}\n\n"
        response += f"📊 **Average per day:** {avg_per_day:.1f} messages"
        
        if avg_per_day > 100:
            response += " - That's a LOT of chatting! 🔥"
        elif avg_per_day > 50:
            response += " - Very active conversation! 💬"
        elif avg_per_day > 10:
            response += " - Steady communication! 👍"
        else:
            response += " - Casual chats! 😊"
        
        return response
    
    def _positive_messages(self) -> str:
        """Answer: Show positive messages"""
        try:
            # Run sentiment analysis if not done
            if 'compound' not in self.df.columns:
                analyzer = SentimentAnalyzer(self.df)
                self.df = analyzer.analyze_vader()
            
            positive_msgs = self.df[self.df['compound'] >= 0.05]
            total_positive = len(positive_msgs)
            percentage = (total_positive / len(self.df)) * 100
            
            response = f"😊 **Positive Messages:** {total_positive:,} ({percentage:.1f}%)\n\n"
            
            if len(positive_msgs) > 0:
                # Show sample positive messages
                samples = positive_msgs.sample(min(3, len(positive_msgs)))
                response += "**Sample positive messages:**\n\n"
                
                for _, msg in samples.iterrows():
                    message_text = msg['message'][:100] + '...' if len(str(msg['message'])) > 100 else msg['message']
                    response += f"- **{msg['sender']}**: {message_text}\n"
                
                response += f"\nYour chat is {percentage:.1f}% positive - that's "
                
                if percentage > 60:
                    response += "really positive! Great vibes! 🌟"
                elif percentage > 40:
                    response += "pretty balanced! Normal conversation mix! 👍"
                else:
                    response += "on the neutral side! That's typical for most chats! 😊"
            else:
                response += "No positive messages found (or sentiment not analyzed yet)."
            
            return response
        except Exception as e:
            return f"Sorry, couldn't analyze sentiment: {str(e)}"
    
    def _negative_messages(self) -> str:
        """Answer: Show negative messages"""
        try:
            # Run sentiment analysis if not done
            if 'compound' not in self.df.columns:
                analyzer = SentimentAnalyzer(self.df)
                self.df = analyzer.analyze_vader()
            
            negative_msgs = self.df[self.df['compound'] <= -0.05]
            total_negative = len(negative_msgs)
            percentage = (total_negative / len(self.df)) * 100
            
            response = f"😔 **Negative Messages:** {total_negative:,} ({percentage:.1f}%)\n\n"
            
            if len(negative_msgs) > 0:
                response += f"That's {percentage:.1f}% of your chat - "
                
                if percentage < 10:
                    response += "very low! Your chat is mostly positive! 🌟"
                elif percentage < 20:
                    response += "pretty normal! Some disagreements are healthy! 👍"
                else:
                    response += "a bit high. Maybe work on keeping things positive? 💭"
            else:
                response += "No negative messages found - your chat is all positive! That's amazing! 🎉"
            
            return response
        except Exception as e:
            return f"Sorry, couldn't analyze sentiment: {str(e)}"
    
    def _top_words(self, participant: str = None) -> str:
        """Answer: Top words used"""
        from collections import Counter
        import re
        
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        if participant:
            messages = self.df[self.df['sender'] == participant]['message'].fillna('').tolist()
            response = f"🔑 **Top words used by {participant}:**\n\n"
        else:
            messages = self.df['message'].fillna('').tolist()
            response = "🔑 **Top words in entire chat:**\n\n"
        
        text = ' '.join(messages).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        word_freq = Counter(words).most_common(10)
        
        if word_freq:
            for i, (word, count) in enumerate(word_freq[:10], 1):
                emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][i-1]
                response += f"{emoji} **{word}**: {count:,} times\n"
        else:
            response += "No words found to analyze."
        
        return response
    
    def _emoji_usage(self) -> str:
        """Answer: Emoji usage"""
        import emoji
        
        all_text = ' '.join(self.df['message'].fillna('').tolist())
        emojis = [c for c in all_text if c in emoji.EMOJI_DATA]
        
        from collections import Counter
        emoji_freq = Counter(emojis).most_common(10)
        
        response = f"😀 **Emoji Usage:**\n\n"
        response += f"Total emojis used: {len(emojis):,}\n\n"
        
        if emoji_freq:
            response += "**Top 10 emojis:**\n\n"
            for i, (emoji_char, count) in enumerate(emoji_freq, 1):
                response += f"{i}. {emoji_char} - {count:,} times\n"
        else:
            response += "No emojis found in your chat!"
        
        return response
    
    def _link_sharing(self) -> str:
        """Answer: Link sharing stats"""
        link_msgs = self.df[self.df['media_type'] == 'link']
        total_links = len(link_msgs)
        
        response = f"🔗 **Link Sharing:**\n\n"
        response += f"Total links shared: {total_links:,}\n\n"
        
        if total_links > 0:
            # Who shares most links
            link_by_sender = link_msgs.groupby('sender').size().sort_values(ascending=False)
            top_link_sharer = link_by_sender.idxmax()
            top_count = link_by_sender.max()
            
            response += f"**Top link sharer:** {top_link_sharer} with {top_count:,} links\n\n"
            
            # Sample links
            response += "**Sample links shared:**\n\n"
            samples = link_msgs.sample(min(5, len(link_msgs)))
            for _, msg in samples.iterrows():
                # Extract URL from message
                url_match = re.search(r'https?://\S+', msg['message'])
                if url_match:
                    url = url_match.group()[:60] + '...' if len(url_match.group()) > 60 else url_match.group()
                    response += f"- {url}\n"
        else:
            response += "No links found in your chat!"
        
        return response
    
    def _first_message(self) -> str:
        """Answer: First message in chat"""
        first_msg = self.df.iloc[0]
        
        response = "📜 **First Message in Chat:**\n\n"
        response += f"**Date:** {first_msg['date']}\n"
        response += f"**From:** {first_msg['sender']}\n"
        response += f"**Message:** {first_msg['message'][:200]}{'...' if len(first_msg['message']) > 200 else ''}\n\n"
        response += "Ah, the beginning of your chat journey! 🚀"
        
        return response
    
    def _last_message(self) -> str:
        """Answer: Last message in chat"""
        last_msg = self.df.iloc[-1]
        
        response = "📱 **Most Recent Message:**\n\n"
        response += f"**Date:** {last_msg['date']}\n"
        response += f"**From:** {last_msg['sender']}\n"
        response += f"**Message:** {last_msg['message'][:200]}{'...' if len(last_msg['message']) > 200 else ''}\n\n"
        response += "Your latest conversation! 💬"
        
        return response
    
    def _average_messages_per_day(self) -> str:
        """Answer: Average messages per day"""
        total = len(self.df)
        days = (pd.to_datetime(self.metadata['date_range']['end']) - pd.to_datetime(self.metadata['date_range']['start'])).days + 1
        avg = total / max(days, 1)
        
        response = f"📊 **Average Messages Per Day:**\n\n"
        response += f"**{avg:.1f} messages/day**\n\n"
        
        if avg > 100:
            response += "That's intense chatting! You're always connected! 🔥📱"
        elif avg > 50:
            response += "Very active! You chat multiple times every day! 💬"
        elif avg > 20:
            response += "Regular communication! Good balance! 👍"
        elif avg > 5:
            response += "Casual but consistent! 😊"
        else:
            response += "Occasional chats! Quality over quantity! 🌟"
        
        return response
    
    def _longest_message(self) -> str:
        """Answer: Longest message"""
        longest_idx = self.df['message_length'].idxmax()
        longest_msg = self.df.loc[longest_idx]
        
        response = "📝 **Longest Message:**\n\n"
        response += f"**From:** {longest_msg['sender']}\n"
        response += f"**Length:** {longest_msg['message_length']:,} characters\n"
        response += f"**Date:** {longest_msg['date']}\n\n"
        response += f"**Message:**\n\n> {longest_msg['message'][:500]}{'...' if len(longest_msg['message']) > 500 else ''}\n\n"
        
        if longest_msg['message_length'] > 1000:
            response += "That's a novel! 📚"
        elif longest_msg['message_length'] > 500:
            response += "Quite a detailed message! 📝"
        else:
            response += "A good lengthy message! 💬"
        
        return response
    
    def _shortest_message(self) -> str:
        """Answer: Shortest message"""
        shortest_idx = self.df['message_length'].idxmin()
        shortest_msg = self.df.loc[shortest_idx]
        
        response = "📱 **Shortest Message:**\n\n"
        response += f"**From:** {shortest_msg['sender']}\n"
        response += f"**Length:** {shortest_msg['message_length']} character(s)\n"
        response += f"**Message:** \"{shortest_msg['message']}\"\n\n"
        
        if shortest_msg['message_length'] <= 2:
            response += "Keeping it brief! 😄"
        else:
            response += "Short and sweet! 💬"
        
        return response
    
    def _who_initiates(self) -> str:
        """Answer: Who initiates conversations"""
        self.df = self.df.sort_values('date')
        self.df['time_diff'] = self.df['date'].diff().dt.total_seconds() / 3600
        
        # Messages after 6+ hour gap
        conversation_starts = self.df[self.df['time_diff'] > 6].copy()
        
        if len(conversation_starts) == 0:
            return "No clear conversation initiators found (chat might be too short)."
        
        initiators = conversation_starts.groupby('sender').size().sort_values(ascending=False)
        top_initiator = initiators.idxmax()
        top_count = initiators.max()
        total = len(conversation_starts)
        percentage = (top_count / total) * 100
        
        response = f"🚀 **Conversation Initiator:**\n\n"
        response += f"**{top_initiator}** starts most conversations with {top_count:,} initiations ({percentage:.1f}%)\n\n"
        
        if percentage > 70:
            response += f"{top_initiator} is definitely the one who keeps the conversation going! 🏆"
        elif percentage > 50:
            response += f"{top_initiator} initiates more often, but it's fairly balanced! ⚖️"
        else:
            response += "You both initiate conversations pretty equally! 🤝"
        
        return response
    
    def _response_time(self) -> str:
        """Answer: Average response time"""
        self.df = self.df.sort_values('date')
        
        response_times = []
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender].copy()
            sender_msgs = sender_msgs.sort_values('date')
            sender_msgs['prev_time'] = sender_msgs['date'].shift(1)
            sender_msgs['response_time'] = (
                sender_msgs['date'] - sender_msgs['prev_time']
            ).dt.total_seconds() / 60
            
            avg_response = sender_msgs['response_time'].median()
            response_times.append({
                'sender': sender,
                'avg_response_time_min': avg_response
            })
        
        response = "⏱️ **Average Response Times:**\n\n"
        
        for rt in response_times:
            avg_time = rt['avg_response_time_min']
            
            if avg_time < 5:
                speed = "⚡ Lightning fast!"
            elif avg_time < 30:
                speed = "🚀 Quick!"
            elif avg_time < 60:
                speed = "💬 Steady"
            else:
                speed = "🐌 Patient"
            
            response += f"**{rt['sender']}**: {avg_time:.1f} minutes - {speed}\n"
        
        return response
    
    def _help(self) -> str:
        """Answer: Help - what questions can I ask?"""
        response = "🤖 **I can help you analyze your chat! Try asking:**\n\n"
        
        questions = [
            ("📊 General Stats", [
                "How many total messages?",
                "What's our average per day?",
                "Who talks more?",
                "When's our peak hour?",
                "What's our most active day?"
            ]),
            ("😊 Sentiment", [
                "Show me positive messages",
                "How many negative messages?",
                "Is our chat positive?"
            ]),
            ("🔑 Words & Emojis", [
                "What words does [name] use most?",
                "What are our top words?",
                "How many emojis do we use?"
            ]),
            ("📱 Messages", [
                "What was the first message?",
                "What's the longest message?",
                "Show me the last message"
            ]),
            ("🚀 Behavior", [
                "Who starts conversations more?",
                "What's our response time?",
                "Who shares more links?"
            ])
        ]
        
        for category, examples in questions:
            response += f"**{category}**\n"
            for q in examples:
                response += f"- {q}\n"
            response += "\n"
        
        response += "💡 **Just type your question naturally and I'll try to answer!**"
        
        return response
    
    def _default_response(self) -> str:
        """Default response for unrecognized questions"""
        return """🤔 Hmm, I'm not sure I understand that question!

Try asking about:
- **Stats**: "How many messages?", "Who talks more?"
- **Timing**: "What's our peak hour?", "Most active day?"
- **Sentiment**: "Show positive messages", "How many negative?"
- **Words**: "Top words?", "What words does [name] use?"
- **Behavior**: "Who initiates?", "Response time?"

Type **"help"** for more examples! 😊"""
