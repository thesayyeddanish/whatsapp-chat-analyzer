import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import your modules
from parsers.chat_parser import WhatsAppChatParser
from analyzers.activity_analyzer import ActivityAnalyzer
from analyzers.participant_analyzer import ParticipantAnalyzer
from sentiment.sentiment_analyzer import SentimentAnalyzer
from visualizers.dashboard import ChatVisualizer

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
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        section = st.sidebar.radio(
            "Select Section",
            ["📊 Overview", "⏰ Activity & Timing", "👥 Participants", 
             "😊 Sentiment", "🏷️ Topics", "📖 Story", "📥 Export"],
            index=0
        )
        
        # ========== OVERVIEW ==========
        if section == "📊 Overview":
            st.header("📊 Chat Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", f"{metadata['total_messages']:,}")
            with col2:
                st.metric("Total Words", f"{metadata['total_words']:,}")
            with col3:
                st.metric("Participants", len(metadata['participants']))
            with col4:
                st.metric("Date Range", f"{metadata['date_range']['start']}")
            
            # Sample messages
            st.subheader("📝 Sample Messages")
            st.dataframe(df[['date', 'sender', 'message']].head(10), use_container_width=True)
            
            # Media breakdown
            if 'media_type' in df.columns:
                st.subheader("📎 Media Type Distribution")
                media_dist = df['media_type'].value_counts()
                st.bar_chart(media_dist)
        
        # ========== ACTIVITY & TIMING ==========
        elif section == "⏰ Activity & Timing":
            st.header("⏰ Activity & Timing Trends")
            
            activity_analyzer = ActivityAnalyzer(df)
            
            # Heatmap
            st.subheader("🔥 Activity Heatmap")
            try:
                heatmap_fig = activity_analyzer.create_heatmap('hourly')
                st.plotly_chart(heatmap_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate heatmap: {str(e)}")
            
            # Active hours
            st.subheader("🕐 Busiest Hours")
            hourly_data, peak = activity_analyzer.get_active_hours()
            st.info(f"🔥 **Peak hour:** {int(peak['hour'])}:00 with {int(peak['message_count'])} messages")
            
            # Active days
            st.subheader("📅 Most Active Days")
            daily_data = activity_analyzer.get_active_days()
            st.bar_chart(daily_data.set_index('day_of_week'))
            
            # Response times
            st.subheader("⏱️ Average Response Times")
            response_times = activity_analyzer.calculate_response_times()
            st.dataframe(response_times, use_container_width=True)
            
            # Initiators
            st.subheader("🚀 Conversation Initiators")
            initiators = activity_analyzer.analyze_initiators()
            st.dataframe(initiators, use_container_width=True)
        
        # ========== PARTICIPANTS ==========
        elif section == "👥 Participants":
            st.header("👥 Participant Insights")
            
            participant_analyzer = ParticipantAnalyzer(df)
            
            # Leaderboard
            st.subheader("🏆 Sender Leaderboard")
            leaderboard = participant_analyzer.get_sender_leaderboard()
            st.dataframe(leaderboard, use_container_width=True)
            
            # Double-texting
            st.subheader("💬 Double-Texting Index")
            double_text = participant_analyzer.calculate_double_texting_index()
            st.dataframe(double_text, use_container_width=True)
            
            # Media ratios
            st.subheader("📸 Media Usage by Participant")
            media_ratios = participant_analyzer.get_media_ratios()
            st.dataframe(media_ratios, use_container_width=True)
            
            # Word clouds
            st.subheader("☁️ Word Clouds")
            vocab_stats = participant_analyzer.get_vocabulary_stats()
            
            selected_user = st.selectbox("Select participant", list(vocab_stats.keys()))
            if selected_user:
                text = ' '.join(df[df['sender'] == selected_user]['message'].fillna('').tolist())
                word_cloud_fig = ChatVisualizer(df).create_word_cloud(text, f"Word Cloud: {selected_user}")
                st.plotly_chart(word_cloud_fig, use_container_width=True)
        
        # ========== SENTIMENT ==========
        elif section == "😊 Sentiment":
            st.header("😊 Sentiment Analysis")
            
            with st.spinner("Running sentiment analysis..."):
                sentiment_analyzer = SentimentAnalyzer(df, use_transformer=False)
                df = sentiment_analyzer.analyze_vader()
            
            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**VADER Sentiment**")
                sentiment_counts = df['vader_sentiment'].value_counts()
                st.write(sentiment_counts)
                st.bar_chart(sentiment_counts)
            
            with col2:
                st.write("**Average Sentiment Scores**")
                avg_scores = {
                    'Positive': df['positive'].mean(),
                    'Neutral': df['neutral'].mean(),
                    'Negative': df['negative'].mean(),
                    'Compound': df['compound'].mean()
                }
                st.write(avg_scores)
            
            # Sentiment timeline
            st.subheader("📈 Sentiment Over Time")
            try:
                sentiment_fig = ChatVisualizer(df).create_sentiment_timeline()
                st.plotly_chart(sentiment_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate timeline: {str(e)}")
            
            # Mood index
            st.subheader("📅 Weekly Mood Index")
            mood_index = sentiment_analyzer.get_mood_index('W')
            st.dataframe(mood_index, use_container_width=True)
        
        # ========== TOPICS ==========
        elif section == "🏷️ Topics":
            st.header("🏷️ Topic Modeling")
            
            st.info("ℹ️ Topic modeling requires additional setup. For now, showing basic keyword analysis.")
            
            # Simple keyword extraction
            st.subheader("🔑 Top Keywords")
            vocab_stats = ParticipantAnalyzer(df).get_vocabulary_stats()
            
            for sender, stats in vocab_stats.items():
                st.write(f"**{sender}**")
                top_words = [word for word, count in stats['top_words'][:10]]
                st.write(", ".join(top_words))
        
        # ========== STORY ==========
        elif section == "📖 Story":
            st.header("📖 Narrative Generator")
            
            st.info("ℹ️ Full narrative generation coming soon. Showing basic chat timeline.")
            
            # Message timeline
            st.subheader("📅 Message Timeline")
            timeline_fig = ChatVisualizer(df).create_message_timeline()
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        # ========== EXPORT ==========
        elif section == "📥 Export":
            st.header("📥 Export & Infographics")
            
            visualizer = ChatVisualizer(df)
            
            # Generate infographic
            st.subheader("🎨 Infographic Card")
            if st.button("Generate Infographic"):
                with st.spinner("Creating infographic..."):
                    stats = {
                        'total_messages': metadata['total_messages'],
                        'total_words': metadata['total_words'],
                        'participants': metadata['participants'],
                        'date_range': metadata['date_range'],
                        'message_count': df.groupby('sender').size().to_dict(),
                        'media_breakdown': metadata.get('media_breakdown', {})
                    }
                    
                    try:
                        img_path = visualizer.create_infographic_card(stats)
                        st.image(img_path, caption="Chat Statistics Infographic")
                        
                        with open(img_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Infographic (PNG)",
                                data=f.read(),
                                file_name='chat_stats.png',
                                mime='image/png'
                            )
                    except Exception as e:
                        st.error(f"Could not generate infographic: {str(e)}")
            
            # Export data
            st.subheader("💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Data (CSV)",
                    data=csv,
                    file_name='chat_analysis.csv',
                    mime='text/csv'
                )
            
            with col2:
                json_data = df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name='chat_analysis.json',
                    mime='application/json'
                )
            
            # Participant comparison chart
            st.subheader("📊 Participant Comparison")
            leaderboard = ParticipantAnalyzer(df).get_sender_leaderboard()
            comparison_fig = ChatVisualizer(df).create_participant_comparison(leaderboard)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.error("""
        **Troubleshooting:**
        - Make sure you exported the chat correctly from WhatsApp
        - Go to: Chat info → Export Chat → **Without Media**
        - The file should be a `.txt` file
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
    5. Save the `.txt` file and upload it here
    
    ### 🚀 Features
    
    - **Activity & Timing** - Heatmaps, peak hours, response times
    - **Participant Insights** - Leaderboards, double-texting, media ratios
    - **Sentiment Analysis** - VADER scoring, mood tracking
    - **Word Clouds** - Most used words and emojis
    - **Export Options** - CSV, JSON, PNG infographics
    """)
