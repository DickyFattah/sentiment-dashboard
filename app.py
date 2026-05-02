import streamlit as st
import pandas as pd
from utils import preprocess_text
from model import load_model, predict_sentiment

# CONFIG
st.set_page_config(page_title="Review Intelligence Dashboard", layout="wide")

# LOAD MODEL
model = load_model()

# HEADER
st.title("📊 Customer Review Intelligence Dashboard")
st.markdown("Analyze customer sentiment & uncover insights instantly")

# UPLOAD DATA
uploaded_file = st.file_uploader("Upload CSV file with reviews", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_reviews.csv")

# PREPROCESS
df['clean'] = df['text'].apply(preprocess_text)
df['sentiment'] = df['clean'].apply(lambda x: predict_sentiment(model, x))

# KPI
col1, col2, col3 = st.columns(3)

total = len(df)
positive = (df['sentiment'] == 'positive').sum()
negative = (df['sentiment'] == 'negative').sum()

col1.metric("Total Reviews", total)
col2.metric("Positive", positive)
col3.metric("Negative", negative)

# CHART
st.subheader("Sentiment Distribution")
st.bar_chart(df['sentiment'].value_counts())

# KEYWORDS
st.subheader("Top Words")
from collections import Counter
words = " ".join(df['clean']).split()
common_words = Counter(words).most_common(10)
st.write(common_words)

# TABLE
st.subheader("Review Data")
st.dataframe(df)