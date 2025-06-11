import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit 기본 설정 ---
st.set_page_config(page_title="HopSkipDrive Review Dashboard", layout="wide")
st.title("AI-Powered HopSkipDrive Review Intelligence Dashboard")
st.markdown("### Overview: Ratings & Sentiment Landscape")

# --- Section 2: Dataset Summary ---
st.write(
    """
    Welcome to the HopSkipDrive Review Analysis Dashboard. This tool helps Product and CX teams **uncover** and 
    **prioritize the most critical user issues** through structured analysis of app store reviews.
    In this dashboard, we analyze reviews from **HopSkipDrive**, examining how user ratings are distributed across 
    dimensions such as **sentiment**, **topics**, and **time** — offering a clear view into the voice of the customer.
    """
)

st.markdown(
    """
    **Don’t have a dataset yet?**  
    No problem — you can generate one instantly using our built-in scraper. It pulls reviews from the 
    Google Play Store and uses **LLM-powered prompt engineering** to classify sentiment and tag user-reported 
    issues, delivering a clean, **ready-to-use CSV file** for this dashboard.
    """
)

st.markdown("---")

# --- Section 3: File Upload ---
st.markdown("### 📁 Upload an App Review Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload an App Review Dataset (CSV)",
    type="csv"
)

DEFAULT_FILE_PATH = "hopskip_subissues.csv"
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_FILE_PATH)
    st.success(f"No file uploaded — using default dataset: `{DEFAULT_FILE_PATH}`")
st.markdown("<br>", unsafe_allow_html=True)

# --- 날짜 처리 ---
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.strftime('%Y-%m')
min_year, max_year = df['year'].min(), df['year'].max()
year_range = f"{min_year} ~ {max_year}"

# --- Sample Reviews ---
st.subheader("Sample Reviews")
st.write(df[["review_id", "review", "rating", "time", "aspect"]].head())

# --- Key Metrics ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Key Metrics from HopSkipDrive Reviews")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Total Reviews Collected", value=len(df))
    st.metric(label="Number of Topics", value=df["topic"].nunique())
with col2:
    st.metric(label="Average Rating", value=round(df["rating"].mean(), 1))
    st.metric(label="Time Range", value=year_range)

# --- Average Rating by Category ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("HopSkipDrive Reviews: Average Rating by Category")
st.write("Below, you can visualize the **average rating** based on different categories such as **sentiment, topic, and year**.")

x_column = st.selectbox("Select a category to analyze average rating", ['sentiment', 'topic', 'year'])

fig, ax = plt.subplots(figsize=(6, 4))
sorted_df = df.groupby(x_column)['rating'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
sorted_df.plot(kind="barh", ax=ax, color=colors)
ax.set_xlabel("Average Rating")
st.pyplot(fig)

# --- 📊 Topic-based Yearly & Monthly Trends ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Visual Trend of HopSkipDrive Review Ratings")
st.write("You can filter the review by **topic** to analyze yearly and monthly trends in average rating.")

selected_topic = st.selectbox("Select a topic value", df["topic"].dropna().unique())
filtered_df = df[df["topic"] == selected_topic]

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Yearly Trend for `{selected_topic}`**")
    fig, ax = plt.subplots(figsize=(6, 4))
    yearly_avg = filtered_df.groupby("year")["rating"].mean().sort_index()
    ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-', color='b')
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    st.pyplot(fig)

with col2:
    st.write(f"**Monthly Trend (latest 6 months)**")
    latest_months = filtered_df["month"].sort_values().unique()[-6:]
    monthly_df = filtered_df[filtered_df["month"].isin(latest_months)]
    monthly_avg = monthly_df.groupby("month")["rating"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='b')
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    st.pyplot(fig)
