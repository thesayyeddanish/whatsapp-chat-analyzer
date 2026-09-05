# 💬 WhatsApp Chat Analyzer

A comprehensive WhatsApp chat analytics dashboard built with Python and Streamlit.

## 🚀 Features

- **Activity & Timing Trends** - Heatmaps, peak hours, response times, initiator analysis
- **Participant Insights** - Message leaderboards, double-texting index, media ratios
- **Sentiment Analysis** - VADER scoring, mood tracking over time
- **Visualizations** - Word clouds, emoji matrices, interactive charts
- **Export Options** - CSV, JSON, PNG infographics

## 🌐 Live Demo

[Deploy your own on Streamlit Cloud](https://share.streamlit.io)

## 📦 Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"

# Run the app
streamlit run app.py
```

## 🌍 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io)**

3. **Click "New app"** or "Create app"

4. **Configure deployment:**
   - **Repository:** `yourusername/whatsapp-chat-analyzer`
   - **Branch:** `main`
   - **Main file path:** `app.py`

5. **Click "Deploy"**

6. **Your app will be live at:** `https://your-app-name.streamlit.app`

## 📁 Project Structure

