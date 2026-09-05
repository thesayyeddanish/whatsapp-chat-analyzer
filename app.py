import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import plotly.express as px

# Import your modules
from parsers.chat_parser import WhatsAppChatParser
from analyzers.activity_analyzer import ActivityAnalyzer
from analyzers.participant_analyzer import ParticipantAnalyzer
from sentiment.sentiment_analyzer import SentimentAnalyzer
from visualizers.dashboard import ChatVisualizer
from utils.chatbot import ChatAnalyzerBot

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 WhatsApp Chat Analyzer")
st.markdown("Upload your WhatsApp chat `.txt` file to get comprehensive insights")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a WhatsApp chat file",
    type=['txt'],
    help="Export chat from WhatsApp: Chat info → Export chat → Without media"
)

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Parse chat
        with st.spinner("📂 Parsing chat file..."):
            parser = WhatsAppChatParser(tmp_path)
            df = parser.parse()
            metadata = parser.get_chat_metadata()
        
        st.success(f"✅ Loaded **{metadata['total_messages']:,}** messages from **{len(metadata['participants'])}** participants")
        
        # Initialize chatbot
        chatbot = ChatAnalyzerBot(df, metadata)
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        section = st.sidebar.radio(
            "Select Section",
            ["📊 Overview", "⏰ Activity & Timing", "👥 Participants", 
             "😊 Sentiment", "🏷️ Topics", "📖 Story", "📥 Export", "🤖 Chatbot"],
            index=0
        )
        
        # ========== OVERVIEW ==========
        if section == "📊 Overview":
            st.header("📊 Chat Overview")
            
            # Welcome message
            st.markdown(f"""
            ### 👋 Welcome to Your Chat Analysis!
            
            This is a comprehensive analysis of your conversation between **{', '.join(metadata['participants'])}**.
            Let's explore what makes your chat unique! 🚀
            """)
            
            # Hero metrics with icons
            st.markdown("### 📈 Quick Stats")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 40px;">💬</div>
                    <div style="font-size: 28px; font-weight: bold; margin-top: 10px;">
                        {messages:,}
                    </div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Total Messages</div>
                </div>
                """.format(messages=metadata['total_messages']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 40px;">📝</div>
                    <div style="font-size: 28px; font-weight: bold; margin-top: 10px;">
                        {words:,}
                    </div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Total Words</div>
                </div>
                """.format(words=metadata['total_words']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 40px;">👥</div>
                    <div style="font-size: 28px; font-weight: bold; margin-top: 10px;">
                        {participants}
                    </div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Participants</div>
                </div>
                """.format(participants=len(metadata['participants'])), unsafe_allow_html=True)
            
            with col4:
                days_count = (pd.to_datetime(metadata['date_range']['end']) - pd.to_datetime(metadata['date_range']['start'])).days + 1
                st.markdown("""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            padding: 20px; border-radius: 15px; text-align: center; color: white;">
                    <div style="font-size: 40px;">📅</div>
                    <div style="font-size: 24px; font-weight: bold; margin-top: 10px;">
                        {days:,}
                    </div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Days of Chat</div>
                </div>
                """.format(days=days_count), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Chat timeline visualization
            st.markdown("### 📅 Your Chat Journey Over Time")
            
            try:
                timeline_fig = ChatVisualizer(df).create_message_timeline()
                timeline_fig.update_layout(
                    height=400,
                    title_x=0.5,
                    showlegend=True,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                st.info("""
                💡 **What you're seeing:** This shows how your conversation evolved over time. 
                Higher peaks mean more active periods - maybe during trips, events, or exciting times!
                """)
            except Exception as e:
                st.warning(f"Could not load timeline: {str(e)}")
            
            st.markdown("---")
            
            # Participant breakdown with fun facts
            st.markdown("### 👥 Who Talks More?")
            
            try:
                leaderboard = ParticipantAnalyzer(df).get_sender_leaderboard()
                
                # Create participant cards
                for idx, row in leaderboard.iterrows():
                    sender = row['sender']
                    msg_count = row['message_count']
                    word_count = row['total_words']
                    avg_length = row['avg_message_length']
                    
                    percentage = (msg_count / len(df)) * 100
                    
                    # Determine talker type
                    if percentage > 60:
                        talker_type = "🏆 The Main Conversationalist"
                    elif percentage > 40:
                        talker_type = "💬 Very Active Participant"
                    else:
                        talker_type = "👍 Thoughtful Contributor"
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; 
                                margin: 10px 0; border-left: 5px solid #667eea;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 24px; font-weight: bold; color: #333;">
                                    👤 {sender}
                                </div>
                                <div style="color: #666; margin-top: 5px;">{talker_type}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 32px; font-weight: bold; color: #667eea;">
                                    {msg_count:,}
                                </div>
                                <div style="color: #999; font-size: 14px;">messages ({percentage:.1f}%)</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 10px;">
                                <div style="font-size: 20px; font-weight: bold; color: #667eea;">
                                    {word_count:,}
                                </div>
                                <div style="font-size: 12px; color: #666;">Total Words</div>
                            </div>
                            <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 10px;">
                                <div style="font-size: 20px; font-weight: bold; color: #f5576c;">
                                    {avg_length:.1f}
                                </div>
                                <div style="font-size: 12px; color: #666;">Avg Length</div>
                            </div>
                            <div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 10px;">
                                <div style="font-size: 20px; font-weight: bold; color: #43e97b;">
                                    {word_count/msg_count:.1f}
                                </div>
                                <div style="font-size: 12px; color: #666;">Words/Msg</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Fun comparison
                if len(leaderboard) >= 2:
                    top_talker = leaderboard.iloc[0]
                    second_talker = leaderboard.iloc[1]
                    ratio = top_talker['message_count'] / second_talker['message_count']
                    
                    if ratio > 2:
                        fun_fact = f"🎯 **Fun Fact:** {top_talker['sender']} sends {ratio:.1f}x more messages than {second_talker['sender']}!"
                    elif ratio > 1.5:
                        fun_fact = f"⚡ **Fun Fact:** {top_talker['sender']} is slightly more talkative with {ratio:.1f}x more messages!"
                    else:
                        fun_fact = f"🤝 **Fun Fact:** You both have pretty balanced conversations! Only {ratio:.1f}x difference."
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                                padding: 20px; border-radius: 15px; margin: 20px 0;">
                        <div style="font-size: 18px; color: #333;">{fun_fact}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.warning(f"Could not analyze participants: {str(e)}")
            
            st.markdown("---")
            
            # Media breakdown
            if 'media_type' in df.columns:
                st.markdown("### 📎 What Are You Sharing?")
                
                media_dist = df['media_type'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    media_fig = px.pie(
                        values=media_dist.values,
                        names=media_dist.index,
                        title='Message Types Distribution',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    media_fig.update_traces(textposition='inside', textinfo='percent+label')
                    media_fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(media_fig, use_container_width=True)
                
                with col2:
                    # Fun facts about media
                    st.markdown("#### 📊 Breakdown:")
                    
                    for media_type, count in media_dist.items():
                        percentage = (count / len(df)) * 100
                        
                        emoji_map = {
                            'text': '💭',
                            'link': '🔗',
                            'media_omitted': '📷',
                            'image': '📸',
                            'video': '🎥',
                            'voice_note': '🎤',
                            'emoji_only': '😀',
                            'location': '📍'
                        }
                        
                        emoji = emoji_map.get(media_type, '📌')
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px; 
                                    background: #f8f9fa; border-radius: 10px;">
                            <div style="font-size: 24px; margin-right: 15px;">{emoji}</div>
                            <div style="flex: 1;">
                                <div style="font-weight: bold; color: #333;">
                                    {media_type.replace('_', ' ').title()}
                                </div>
                                <div style="color: #666; font-size: 14px;">
                                    {count:,} messages ({percentage:.1f}%)
                                </div>
                            </div>
                            <div style="width: 100px;">
                                <div style="background: #e0e0e0; border-radius: 10px; height: 8px;">
                                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); 
                                                width: {min(percentage, 100)}%; height: 100%; 
                                                border-radius: 10px;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.info("""
                💡 **Tip:** High media sharing means you're sharing lots of moments together! 
                Photos, videos, and links make conversations more engaging.
                """)
            
            st.markdown("---")
            
            # Sample messages preview
            st.markdown("### 💬 Recent Messages Preview")
            
            with st.expander("📝 See what your chat looks like (first 10 messages)"):
                st.dataframe(
                    df[['date', 'sender', 'message']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.info("""
                **What you're seeing:**
                - **date**: When the message was sent
                - **sender**: Who sent it
                - **message**: The actual content
                
                This is how our system organizes your chat for analysis!
                """)
            
            # Next steps
            st.markdown("""
            ### 🎯 What's Next?
            
            Use the sidebar to explore different aspects of your chat:
            
            - ⏰ **Activity & Timing** - See when you chat most (heatmaps, peak hours!)
            - 👥 **Participants** - Deep dive into who talks how much
            - 😊 **Sentiment** - Is your chat positive or negative?
            - 🏷️ **Topics** - What do you talk about most?
            - 📖 **Story** - Visual timeline of your conversation
            - 📥 **Export** - Download beautiful infographics to share!
            - 🤖 **Chatbot** - Ask questions about your chat!
            
            **Happy exploring! 🎉**
            """)
        
        # ========== CHATBOT ==========
        elif section == "🤖 Chatbot":
            st.header("🤖 Chat Analyzer Bot")
            
            st.markdown("""
            ### 💬 Ask Me Anything About Your Chat!
            
            I'm your personal chat assistant. Ask me questions and I'll analyze your conversation!
            """)
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div style="font-weight: bold; color: #1976d2; margin-bottom: 5px;">👤 You</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div style="font-weight: bold; color: #388e3c; margin-bottom: 5px;">🤖 Bot</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chat input
            st.markdown("---")
            
            # Quick question buttons
            st.markdown("#### ⚡ Quick Questions:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Who talks more?", key="q1"):
                    st.session_state.chat_history.append({"role": "user", "content": "Who talks more?"})
                    answer = chatbot.answer_question("Who talks more?")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            with col2:
                if st.button("🕐 Peak hour?", key="q2"):
                    st.session_state.chat_history.append({"role": "user", "content": "What's our peak hour?"})
                    answer = chatbot.answer_question("What's our peak hour?")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            with col3:
                if st.button("😊 Positive messages?", key="q3"):
                    st.session_state.chat_history.append({"role": "user", "content": "Show positive messages"})
                    answer = chatbot.answer_question("Show positive messages")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            # Text input
            user_input = st.text_input(
                "Type your question here:",
                placeholder="e.g., 'How many messages do we send per day?' or 'Who initiates more?'",
                key="chat_input"
            )
            
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get bot response
                with st.spinner("🤔 Thinking..."):
                    answer = chatbot.answer_question(user_input)
                
                # Add bot response to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Rerun to show new message
                st.rerun()
            
            # Clear chat button
            st.markdown("---")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Help section
            with st.expander("❓ What can I ask?"):
                st.markdown("""
                **Try these questions:**
                
                **📊 General Stats:**
                - How many total messages?
                - What's our average per day?
                - Who talks more?
                - When's our peak hour?
                - What's our most active day?
                
                **😊 Sentiment:**
                - Show me positive messages
                - How many negative messages?
                - Is our chat positive?
                
                **🔑 Words & Emojis:**
                - What words does [name] use most?
                - What are our top words?
                - How many emojis do we use?
                
                **📱 Messages:**
                - What was the first message?
                - What's the longest message?
                - Show me the last message
                
                **🚀 Behavior:**
                - Who starts conversations more?
                - What's our response time?
                - Who shares more links?
                
                Just type naturally and I'll do my best to answer! 😊
                """)
        
        # ========== ACTIVITY & TIMING ==========
        elif section == "⏰ Activity & Timing":
            st.header("⏰ Activity & Timing Trends")
            
            st.markdown("""
            ### 🕐 When Do You Chat the Most?
            
            Discover your conversation patterns - are you night owls or early birds?
            """)
            
            activity_analyzer = ActivityAnalyzer(df)
            
            # Heatmap
            st.subheader("🔥 Activity Heatmap")
            try:
                heatmap_fig = activity_analyzer.create_heatmap('hourly')
                heatmap_fig.update_layout(
                    height=500,
                    title_x=0.5,
                    template='plotly_white'
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                st.info("""
                💡 **Reading the heatmap:**
                - **Darker colors** = More messages at that time
                - **X-axis**: Hour of day (0-23)
                - **Y-axis**: Day of week
                - Look for the darkest spots to find your peak chat times!
                """)
            except Exception as e:
                st.warning(f"Could not generate heatmap: {str(e)}")
            
            st.markdown("---")
            
            # Active hours
            st.subheader("🕐 Busiest Hours of the Day")
            try:
                hourly_data, peak = activity_analyzer.get_active_hours()
                
                # Create hour labels
                hourly_data['hour_label'] = hourly_data['hour'].apply(
                    lambda x: f"{x:02d}:00"
                )
                
                # Bar chart
                hour_fig = px.bar(
                    hourly_data,
                    x='hour_label',
                    y='message_count',
                    title='Messages by Hour of Day',
                    labels={'hour_label': 'Hour', 'message_count': 'Messages'},
                    color='message_count',
                    color_continuous_scale='YlOrRd'
                )
                hour_fig.update_layout(
                    height=400,
                    xaxis_title='Hour of Day',
                    yaxis_title='Number of Messages',
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(hour_fig, use_container_width=True)
                
                # Peak hour highlight
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 15px; margin: 20px 0; color: white;">
                    <div style="font-size: 24px; font-weight: bold; text-align: center;">
                        🏆 Peak Hour: {int(peak['hour']):02d}:00
                    </div>
                    <div style="font-size: 18px; text-align: center; margin-top: 10px;">
                        {int(peak['message_count']):,} messages sent during this hour
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("""
                💡 **What this tells you:** This shows which hours you're most active. 
                Maybe you chat during lunch breaks, commute, or before bed!
                """)
            except Exception as e:
                st.warning(f"Could not analyze hourly activity: {str(e)}")
            
            st.markdown("---")
            
            # Active days
            st.subheader("📅 Most Active Days of the Week")
            try:
                daily_data = activity_analyzer.get_active_days()
                
                # Bar chart
                day_fig = px.bar(
                    daily_data,
                    x='day_of_week',
                    y='count',
                    title='Messages by Day of Week',
                    labels={'day_of_week': 'Day', 'count': 'Messages'},
                    color='count',
                    color_continuous_scale='Blues'
                )
                day_fig.update_layout(
                    height=400,
                    xaxis_title='Day of Week',
                    yaxis_title='Number of Messages',
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(day_fig, use_container_width=True)
                
                # Most active day
                most_active_day = daily_data.iloc[0]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 20px; border-radius: 15px; margin: 20px 0; color: white;">
                    <div style="font-size: 24px; font-weight: bold; text-align: center;">
                        📅 Most Active: {most_active_day['day_of_week']}
                    </div>
                    <div style="font-size: 18px; text-align: center; margin-top: 10px;">
                        {most_active_day['count']:,} messages on average
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not analyze daily activity: {str(e)}")
            
            st.markdown("---")
            
            # Response times
            st.subheader("⏱️ Average Response Times")
            try:
                response_times = activity_analyzer.calculate_response_times()
                
                if len(response_times) > 0:
                    # Create response time cards
                    for _, row in response_times.iterrows():
                        avg_time = row['avg_response_time_min']
                        
                        if avg_time < 5:
                            speed_emoji = "⚡ Lightning Fast"
                            speed_desc = "You reply almost instantly!"
                        elif avg_time < 30:
                            speed_emoji = "🚀 Quick Responder"
                            speed_desc = "You reply within minutes"
                        elif avg_time < 60:
                            speed_emoji = "💬 Steady Pace"
                            speed_desc = "You reply within an hour"
                        else:
                            speed_emoji = "🐌 Patient"
                            speed_desc = "You take your time to reply"
                        
                        st.markdown(f"""
                        <div style="background: white; padding: 15px; border-radius: 10px; 
                                    margin: 10px 0; border-left: 4px solid #4facfe;
                                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-size: 20px; font-weight: bold;">
                                        👤 {row['sender']}
                                    </div>
                                    <div style="color: #666; font-size: 14px;">{speed_emoji}</div>
                                    <div style="color: #999; font-size: 12px;">{speed_desc}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 32px; font-weight: bold; color: #4facfe;">
                                        {avg_time:.1f}
                                    </div>
                                    <div style="color: #999; font-size: 14px;">minutes avg</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("""
                    💡 **What is response time?** This measures how long each person takes to reply 
                    after the other person sends a message. Lower is faster!
                    """)
                else:
                    st.info("Response time data not available")
            except Exception as e:
                st.warning(f"Could not calculate response times: {str(e)}")
            
            st.markdown("---")
            
            # Initiators
            st.subheader("🚀 Who Starts Conversations?")
            try:
                initiators = activity_analyzer.analyze_initiators()
                
                if len(initiators) > 0:
                    # Pie chart
                    init_fig = px.pie(
                        initiators,
                        values='conversations_started',
                        names='sender',
                        title='Who Initiates More Conversations?',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    init_fig.update_traces(textposition='inside', textinfo='percent+label')
                    init_fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(init_fig, use_container_width=True)
                    
                    # Top initiator
                    top_initiator = initiators.iloc[0]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 15px; margin: 20px 0; color: white;">
                        <div style="font-size: 24px; font-weight: bold; text-align: center;">
                            🎯 Top Initiator: {top_initiator['sender']}
                        </div>
                        <div style="font-size: 18px; text-align: center; margin-top: 10px;">
                            Starts {top_initiator['conversations_started']} conversations ({top_initiator['percentage']:.1f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("""
                    💡 **What is an initiator?** This counts who sends the first message after a long gap 
                    (6+ hours). It shows who's more likely to start a conversation!
                    """)
                else:
                    st.info("Initiator data not available")
            except Exception as e:
                st.warning(f"Could not analyze initiators: {str(e)}")
        
        # ========== PARTICIPANTS ==========
        elif section == "👥 Participants":
            st.header("👥 Participant Insights")
            
            st.markdown("""
            ### 📊 Deep Dive into Each Person's Chat Style
            
            Who talks more? Who uses more emojis? Let's find out!
            """)
            
            participant_analyzer = ParticipantAnalyzer(df)
            
            # Leaderboard
            st.subheader("🏆 Message Leaderboard")
            try:
                leaderboard = participant_analyzer.get_sender_leaderboard()
                
                # Create a nice table
                st.dataframe(
                    leaderboard,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "sender": "👤 Participant",
                        "message_count": "💬 Messages",
                        "total_words": "📝 Total Words",
                        "avg_message_length": "📏 Avg Length",
                        "avg_words_per_message": "📊 Words/Msg"
                    }
                )
                
                # Visual comparison
                participant_fig = px.bar(
                    leaderboard,
                    x='sender',
                    y='message_count',
                    title='Messages per Participant',
                    labels={'sender': 'Participant', 'message_count': 'Messages'},
                    color='message_count',
                    color_continuous_scale='Viridis'
                )
                participant_fig.update_layout(
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(participant_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load leaderboard: {str(e)}")
            
            st.markdown("---")
            
            # Double-texting
            st.subheader("💬 Double-Texting Index")
            try:
                double_text = participant_analyzer.calculate_double_texting_index()
                
                if len(double_text) > 0:
                    # Explanation
                    st.markdown("""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 15px 0;">
                        <div style="font-size: 16px; color: #1976d2;">
                            <strong>💡 What is double-texting?</strong><br>
                            This shows who sends multiple messages in a row before getting a reply.
                            Higher numbers mean they're more likely to send consecutive messages!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visual
                    dt_fig = px.bar(
                        double_text,
                        x='sender',
                        y='consecutive_messages',
                        title='Consecutive Messages Before Reply',
                        labels={'sender': 'Participant', 'consecutive_messages': 'Consecutive Messages'},
                        color='consecutive_messages',
                        color_continuous_scale='Oranges'
                    )
                    dt_fig.update_layout(
                        height=400,
                        showlegend=False,
                        template='plotly_white'
                    )
                    st.plotly_chart(dt_fig, use_container_width=True)
                    
                    # Table
                    st.dataframe(
                        double_text,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "sender": "👤 Participant",
                            "consecutive_messages": "📱 Consecutive Messages",
                            "double_text_ratio": "📊 Ratio"
                        }
                    )
                else:
                    st.info("Double-texting data not available")
            except Exception as e:
                st.warning(f"Could not analyze double-texting: {str(e)}")
            
            st.markdown("---")
            
            # Media ratios
            st.subheader("📸 Media Usage by Participant")
            try:
                media_ratios = participant_analyzer.get_media_ratios()
                
                if len(media_ratios) > 0:
                    st.dataframe(
                        media_ratios,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("""
                    💡 **What you're seeing:** This shows what types of media each person shares most.
                    Look for who shares the most links, photos, or voice notes!
                    """)
                else:
                    st.info("No media type data available")
            except Exception as e:
                st.warning(f"Could not analyze media usage: {str(e)}")
            
            st.markdown("---")
            
            # Word clouds
            st.subheader("☁️ Word Clouds")
            try:
                vocab_stats = participant_analyzer.get_vocabulary_stats()
                
                selected_user = st.selectbox(
                    "Select participant", 
                    list(vocab_stats.keys()),
                    key="wordcloud_user"
                )
                
                if selected_user:
                    text = ' '.join(df[df['sender'] == selected_user]['message'].fillna('').tolist())
                    
                    if text.strip():
                        word_cloud_fig = ChatVisualizer(df).create_word_cloud(text, f"Word Cloud: {selected_user}")
                        word_cloud_fig.update_layout(height=500)
                        st.plotly_chart(word_cloud_fig, use_container_width=True)
                        
                        st.info(f"""
                        💡 **Word cloud for {selected_user}:**
                        Bigger words = used more frequently. This shows their most common words!
                        """)
                    else:
                        st.warning("No text available for word cloud")
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
        
        # ========== SENTIMENT ==========
        elif section == "😊 Sentiment":
            st.header("😊 Sentiment Analysis")
            
            st.markdown("""
            ### 😊 Is Your Chat Positive or Negative?
            
            Discover the emotional tone of your conversations over time!
            """)
            
            with st.spinner("Running sentiment analysis..."):
                try:
                    sentiment_analyzer = SentimentAnalyzer(df, use_transformer=False)
                    df = sentiment_analyzer.analyze_vader()
                    st.success("✅ Sentiment analysis complete!")
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {str(e)}")
                    # Create default columns
                    df['positive'] = 0.0
                    df['neutral'] = 1.0
                    df['negative'] = 0.0
                    df['compound'] = 0.0
                    df['vader_sentiment'] = 'neutral'
            
            # Sentiment distribution
            st.subheader("📊 Overall Sentiment Breakdown")
            
            if 'vader_sentiment' in df.columns:
                sentiment_counts = df['vader_sentiment'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    sent_fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title='Message Sentiment Distribution',
                        hole=0.4,
                        color_discrete_sequence=['#43e97b', '#ffa07a', '#ff6b6b']
                    )
                    sent_fig.update_traces(textposition='inside', textinfo='percent+label')
                    sent_fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(sent_fig, use_container_width=True)
                
                with col2:
                    # Stats
                    st.markdown("#### 📈 Sentiment Stats:")
                    
                    total = len(df)
                    for sentiment, count in sentiment_counts.items():
                        pct = (count / total) * 100
                        
                        emoji_map = {
                            'positive': '😊',
                            'neutral': '😐',
                            'negative': '😔'
                        }
                        
                        emoji = emoji_map.get(sentiment, '📊')
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 15px 0; padding: 15px; 
                                    background: #f8f9fa; border-radius: 10px;">
                            <div style="font-size: 32px; margin-right: 15px;">{emoji}</div>
                            <div style="flex: 1;">
                                <div style="font-size: 18px; font-weight: bold; color: #333;">
                                    {sentiment.title()}
                                </div>
                                <div style="color: #666; font-size: 14px;">
                                    {count:,} messages
                                </div>
                            </div>
                            <div style="font-size: 24px; font-weight: bold; color: #667eea;">
                                {pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("""
                    💡 **What is sentiment analysis?**
                    - **Positive**: Happy, excited, loving messages
                    - **Neutral**: Normal, factual conversations
                    - **Negative**: Sad, angry, or frustrated messages
                    
                    Most chats are mostly neutral with some positive moments!
                    """)
            else:
                st.write("No sentiment data available")
            
            st.markdown("---")
            
            # Sentiment timeline
            st.subheader("📈 Sentiment Over Time")
            try:
                if 'compound' in df.columns:
                    sentiment_fig = ChatVisualizer(df).create_sentiment_timeline()
                    sentiment_fig.update_layout(
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    st.info("""
                    💡 **Reading the sentiment timeline:**
                    - **Above 0.05** (green line): Positive mood
                    - **Below -0.05** (red line): Negative mood
                    - **In between**: Neutral conversation
                    
                    Look for patterns - maybe weekends are more positive!
                    """)
                else:
                    st.warning("Sentiment data not available")
            except Exception as e:
                st.warning(f"Could not generate timeline: {str(e)}")
            
            st.markdown("---")
            
            # Mood index
            st.subheader("📅 Weekly Mood Index")
            try:
                if all(col in df.columns for col in ['compound', 'positive', 'negative', 'neutral']):
                    mood_index = sentiment_analyzer.get_mood_index('W')
                    
                    if len(mood_index) > 0:
                        st.dataframe(
                            mood_index,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.info("""
                        💡 **Weekly Mood Index:**
                        This shows the average sentiment for each week.
                        Higher compound scores = more positive weeks!
                        """)
                    else:
                        st.info("No mood data available")
                else:
                    st.warning("Sentiment data not available for mood index")
            except Exception as e:
                st.warning(f"Could not generate mood index: {str(e)}")
        
        # ========== TOPICS ==========
        elif section == "🏷️ Topics":
            st.header("🏷️ Topic Modeling")
            
            st.info("ℹ️ Topic modeling requires additional setup. For now, showing basic keyword analysis.")
            
            # Simple keyword extraction
            st.subheader("🔑 Top Keywords")
            try:
                vocab_stats = ParticipantAnalyzer(df).get_vocabulary_stats()
                
                for sender, stats in vocab_stats.items():
                    st.markdown(f"#### 👤 {sender}")
                    
                    if stats['top_words']:
                        # Create word tags
                        words_html = ""
                        for word, count in stats['top_words'][:15]:
                            words_html += f"""
                            <span style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        color: white; padding: 8px 15px; border-radius: 20px; 
                                        margin: 5px; font-size: 14px; font-weight: bold;">
                                {word} ({count})
                            </span>
                            """
                        
                        st.markdown(f"""
                        <div style="margin: 15px 0;">{words_html}</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("No keywords available")
                    
                    st.markdown("---")
                    
            except Exception as e:
                st.warning(f"Could not extract keywords: {str(e)}")
        
        # ========== STORY ==========
        elif section == "📖 Story":
            st.header("📖 Narrative Generator")
            
            st.info("ℹ️ Full narrative generation coming soon. Showing basic chat timeline.")
            
            # Message timeline
            st.subheader("📅 Message Timeline")
            try:
                timeline_fig = ChatVisualizer(df).create_message_timeline()
                timeline_fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                st.info("""
                💡 **Your Chat Story:**
                This shows the complete journey of your conversation from start to finish.
                Each peak represents a busy period - maybe a trip, event, or exciting news!
                """)
            except Exception as e:
                st.warning(f"Could not generate timeline: {str(e)}")
        
        # ========== EXPORT ==========
        elif section == "📥 Export":
            st.header("📥 Export & Infographics")
            
            st.markdown("""
            ### 🎨 Create Beautiful Infographics to Share!
            
            Download your chat stats as images or data files.
            """)
            
            visualizer = ChatVisualizer(df)
            
            # Generate infographic
            st.subheader("🎨 Infographic Card")
            if st.button("Generate Infographic", type="primary"):
                with st.spinner("Creating infographic..."):
                    try:
                        stats = {
                            'total_messages': metadata['total_messages'],
                            'total_words': metadata['total_words'],
                            'participants': metadata['participants'],
                            'date_range': metadata['date_range'],
                            'message_count': df.groupby('sender').size().to_dict(),
                            'media_breakdown': metadata.get('media_breakdown', {})
                        }
                        
                        img_path = visualizer.create_infographic_card(stats)
                        st.image(img_path, caption="Chat Statistics Infographic", use_column_width=True)
                        
                        with open(img_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Infographic (PNG)",
                                data=f.read(),
                                file_name='chat_stats.png',
                                mime='image/png',
                                type="primary"
                            )
                        
                        st.success("✅ Infographic created! Click the download button above to save it.")
                        
                    except Exception as e:
                        st.error(f"Could not generate infographic: {str(e)}")
            
            st.markdown("---")
            
            # Export data
            st.subheader("💾 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name='chat_analysis.csv',
                    mime='text/csv',
                    type="primary"
                )
                st.info("Best for Excel, Google Sheets")
            
            with col2:
                json_data = df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name='chat_analysis.json',
                    mime='application/json',
                    type="primary"
                )
                st.info("Best for developers, apps")
            
            with col3:
                markdown_text = f"""
# Chat Analysis Summary

- **Total Messages:** {metadata['total_messages']:,}
- **Total Words:** {metadata['total_words']:,}
- **Participants:** {', '.join(metadata['participants'])}
- **Date Range:** {metadata['date_range']['start']} to {metadata['date_range']['end']}
- **Days Active:** {(pd.to_datetime(metadata['date_range']['end']) - pd.to_datetime(metadata['date_range']['start'])).days + 1:,}
                """
                st.download_button(
                    label="📝 Download Summary",
                    data=markdown_text,
                    file_name='chat_summary.md',
                    mime='text/markdown',
                    type="primary"
                )
                st.info("Best for notes, docs")
            
            st.markdown("---")
            
            # Participant comparison chart
            st.subheader("📊 Participant Comparison")
            try:
                leaderboard = ParticipantAnalyzer(df).get_sender_leaderboard()
                comparison_fig = ChatVisualizer(df).create_participant_comparison(leaderboard)
                comparison_fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate comparison chart: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.error("""
        **How to Export WhatsApp Chat Correctly:**
        
        **On Android:**
        1. Open the chat
        2. Tap the three dots (⋮) → More → Export chat
        3. Choose **WITHOUT MEDIA**
        4. Save the .txt file
        
        **On iPhone:**
        1. Open the chat
        2. Tap the contact/group name at top
        3. Scroll down → Export Chat
        4. Choose **Without Media**
        5. Save to Files
        
        **File should look like:**
        ```
        12/31/23, 11:59 PM - John Doe: Hey!
        1/1/24, 12:00 AM - Jane Smith: Happy New Year!
        ```
        """)
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Upload a WhatsApp chat file to begin analysis")
    
    st.markdown("""
    ### 📱 How to Export WhatsApp Chat:
    
    1. Open the chat in WhatsApp (individual or group)
    2. Tap on contact/group name at top
    3. Scroll down and tap **Export Chat**
    4. Choose **Without Media** (recommended for faster analysis)
    5. Upload the `.txt` file to the app
    
    ### 🚀 Features
    
    - **Activity & Timing** - Heatmaps, peak hours, response times
    - **Participant Insights** - Leaderboards, double-texting, media ratios
    - **Sentiment Analysis** - VADER scoring, mood tracking
    - **Word Clouds** - Most used words and emojis
    - **Export Options** - CSV, JSON, PNG infographics
    - **🤖 Chatbot** - Ask questions about your chat!
    
    ### 💡 Tips
    
    - Export chats **without media** for faster analysis
    - Works with both individual and group chats
    - Supports chats from 2021 onwards
    - All analysis happens in your browser - your data is private!
    """)
