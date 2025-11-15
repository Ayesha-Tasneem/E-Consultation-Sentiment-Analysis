Interactive Streamlit dashboard for VADER-based sentiment analysis with charts and CSV export.
# E-Consultation Sentiment Analysis (Streamlit)
A modern, fully interactive sentiment analysis dashboard built using **Streamlit**, **NLTK VADER**, and **Plotly**.  
This app allows healthcare professionals or analysts to upload consultation feedback, automatically detect sentiment, visualize insights, and export results — all in seconds.
This project features a dark, elegant UI and provides deep insights into patient sentiment trends.

## Features

### **1. Automated Sentiment Analysis**
- Uses **NLTK VADER** for polarity scoring  
- Classifies sentiment into:
  - Positive  
  - Neutral  
  - Negative  
- Adds a compound sentiment score for each comment  

### **2. Visual Analytics**
Includes beautiful, interactive charts powered by Plotly:
- Bar chart (sentiment distribution)  
- Pie chart (sentiment ratio)  
- Trend line over time (if date column exists)  

### **3. CSV Upload & Smart Column Detection**
- Upload any CSV file  
- Select the column containing comments  
- Handles corrupted rows or edge cases  

### **4. Data Cleaning & Preview**
- View top 100 rows  
- View top positive and negative comments  
- Auto-ignore invalid or empty rows  

### **5. Insights Dashboard**
- Summary metrics  
- Sentiment counts  
- Top comments sorted by score  
- Optional time-based trend (if CSV has a date column)  

### **6. Export Results**
- Download a processed CSV with sentiment scores & labels

## UI & UX
This project uses a **custom dark theme** with:
- Smooth UI  
- Elegant typography  
- Custom color palette  
- Minimalistic but premium layout  

## Tech Stack

- Python
- Streamlit
- NLTK (VADER Sentiment Analyzer)
- Pandas
- Plotly
- Custom CSS styling
