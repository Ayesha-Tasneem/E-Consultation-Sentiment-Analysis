# app.py
import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# Page setup
st.set_page_config(
   page_title="E-Consultation Sentiment Analysis",
   layout="centered"
)

# Dark minimal styling
st.markdown(
   """
   <style>
   :root {
       --primary: #111111;
       --primary-dark: #000000;
       --text: #FFFFFF;
       --muted: #D1D5DB;
       --bg-card: #0B0B0B;
       --bg-soft: #000000;
       --accent-1: #1F2937;
       --pos: #16A34A;
       --neu: #F59E0B;
       --neg: #EF4444;
   }
   .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px; }
   h1, h2, h3 { color: var(--text); }
   .stMetric label { color: var(--muted) !important; }
   div[data-testid="stMetricValue"] { color: var(--text); }
   .stDownloadButton button, .stButton>button {
       background: var(--primary); color: white;
       border-radius: 8px; border: 1px solid var(--text);
   }
   .stDownloadButton button:hover, .stButton>button:hover { background: var(--primary-dark); }
   body, .block-container { background-color: var(--bg-soft); color: var(--text); }
   </style>
   """,
   unsafe_allow_html=True
)

# Ensure VADER
def ensure_vader():
   if not st.session_state.get('_vader_ready'):
       try:
           from nltk.data import find
           try:
               find('sentiment/vader_lexicon.zip')
           except LookupError:
               nltk.download('vader_lexicon', quiet=True)
       finally:
           st.session_state['_vader_ready'] = True

ensure_vader()

# Title
st.title("📊 E-Consultation Sentiment Analysis")
st.caption("Analyze feedback sentiment, visualize results, and export findings.")

# Upload directly on homepage
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_csv(file_bytes):
   from io import BytesIO
   bio = BytesIO(file_bytes)
   return pd.read_csv(bio, engine='python', on_bad_lines='skip')

if uploaded_file:
   try:
       file_bytes = uploaded_file.getvalue()
       if not file_bytes:
           st.error("Uploaded file is empty.")
           st.stop()
       df = load_csv(file_bytes)
   except Exception:
       st.error("Could not parse the uploaded CSV. Please check the file.")
       st.stop()

   # Select column
   text_col = st.selectbox("Select the column with comments", df.columns)

   if text_col:
       # Sentiment analysis
       sia = SentimentIntensityAnalyzer()
       df['Sentiment Score'] = df[text_col].astype(str).apply(lambda x: sia.polarity_scores(str(x))['compound'])

       def get_sentiment_label(c):
           if c >= 0.05:
               return 'Positive'
           elif c <= -0.05:
               return 'Negative'
           else:
               return 'Neutral'

       df['Sentiment'] = df['Sentiment Score'].apply(get_sentiment_label)
       counts = df['Sentiment'].value_counts()
       total = counts.sum()

       # Show metrics inline
       col1, col2, col3 = st.columns(3)
       col1.metric("Positive", counts.get('Positive', 0))
       col2.metric("Neutral", counts.get('Neutral', 0))
       col3.metric("Negative", counts.get('Negative', 0))
       st.caption(f"Total comments: {total}")

       # Tabs
       tabs = st.tabs(["Results", "Preview", "Insights", "Download"])

       # RESULTS Tab
       with tabs[0]:
           st.subheader("Sentiment Distribution")
           col1, col2 = st.columns(2)

           with col1:
               fig_bar = px.bar(
                   x=['Positive', 'Neutral', 'Negative'],
                   y=[counts.get('Positive', 0), counts.get('Neutral', 0), counts.get('Negative', 0)],
                   color=['Positive', 'Neutral', 'Negative'],
                   color_discrete_map={"Positive": "#16A34A", "Neutral": "#F59E0B", "Negative": "#EF4444"},
                   labels={"x": "Sentiment", "y": "Count"}
               )
               st.plotly_chart(fig_bar, use_container_width=True)

           with col2:
               fig_pie = px.pie(
                   values=[counts.get('Positive', 0), counts.get('Neutral', 0), counts.get('Negative', 0)],
                   names=['Positive', 'Neutral', 'Negative'],
                   color=['Positive', 'Neutral', 'Negative'],
                   color_discrete_map={"Positive": "#16A34A", "Neutral": "#F59E0B", "Negative": "#EF4444"}
               )
               st.plotly_chart(fig_pie, use_container_width=True)

           st.subheader("Sample Comments")
           st.dataframe(df[[text_col, 'Sentiment Score', 'Sentiment']].head(20))

       # PREVIEW Tab
       with tabs[1]:
           st.subheader("Data Preview")
           st.dataframe(df.head(100))

       # INSIGHTS Tab
       with tabs[2]:
           st.subheader("Top Comments")
           st.markdown("**Top Positive Comments**")
           st.dataframe(df.sort_values('Sentiment Score', ascending=False).head(5)[[text_col, 'Sentiment Score']])
           st.markdown("**Top Negative Comments**")
           st.dataframe(df.sort_values('Sentiment Score', ascending=True).head(5)[[text_col, 'Sentiment Score']])

           date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
           if date_cols:
               st.subheader("Sentiment Trend Over Time")
               date_col = st.selectbox("Select date column", options=date_cols)
               df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
               trend = df.groupby([df['date_parsed'].dt.date, 'Sentiment']).size().unstack(fill_value=0)
               fig_line = px.line(
                   trend,
                   x=trend.index,
                   y=['Positive', 'Neutral', 'Negative'],
                   color_discrete_map={"Positive": "#16A34A", "Neutral": "#F59E0B", "Negative": "#EF4444"}
               )
               st.plotly_chart(fig_line, use_container_width=True)
           else:
               st.info("Add a 'date' column to see the sentiment trend.")

       # DOWNLOAD Tab
       with tabs[3]:
           st.subheader("Export Results")
           csv = df.to_csv(index=False).encode('utf-8')
           st.download_button(
               "Download CSV with Sentiment",
               data=csv,
               file_name="sentiment_results.csv",
               mime="text/csv"
           )

else:
   st.info("👆 Upload a CSV to get started.")
