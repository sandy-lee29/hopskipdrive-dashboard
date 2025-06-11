# --- Streamlit App with Top Issue and Toggle Sub Issues (Improved Version) ---
import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import numpy as np
import seaborn as sns
import os

# --- Initialize session state for safe access ---
if 'top_issue_summary_all' not in st.session_state:
    st.session_state['top_issue_summary_all'] = pd.DataFrame()
if 'topic_summary' not in st.session_state:
    st.session_state['topic_summary'] = pd.DataFrame()

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("HopSkipDrive Review Intelligence Dashboard")
st.markdown("#### 🎯 Company Insights: Ranked Issues & Business Impact")

# --- OpenAI API Key ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

client = openai.Client(api_key=OPENAI_API_KEY)

st.markdown("""
This dashboard analyzes HopSkipDrive user reviews to **uncover the most impactful issues affecting ride experience** and **potential revenue**.
We estimate **churn rate** and **revenue impact score** using the proportion of negative reviews and their frequency.
These metrics are designed to help **Product and CX teams prioritize issues** that matter most for improving our ride-sharing service.
""")
st.markdown("---")

# Step 1: Upload file or use default
def load_data():
    uploaded_file = st.file_uploader("Upload your review CSV file", type=["csv"])
    DEFAULT_FILE = "hopskip_subissues.csv"
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    else:
        try:
            df = pd.read_csv(DEFAULT_FILE)
            st.info(f" Using default file: `{DEFAULT_FILE}`")
        except FileNotFoundError:
            st.error(f"Default file `{DEFAULT_FILE}` not found. Please upload a CSV file to continue.")
            st.stop()
    return df

df = load_data()

# Preview the data
st.markdown("Preview of the Dataset")
st.dataframe(df.head(3))
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# Step 2: Topic-Level Feature Importance Analysis
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.subheader("Step 2. Rank Topics by Estimated Revenue Risk")

st.markdown("""
Click the button below to estimate potential **churn risk** and **revenue impact** based on review data.

#### 📊 Risk Score Methodology
- **Churn Risk**: Calculated from negative review % with small variation to account for uncertainty
- **Revenue Risk**: Combines churn risk with review volume to estimate business impact
- **Reliability**: Requires minimum 3 reviews, ensures scores stay within realistic bounds
""")

if st.button("Create Proxy Metrics"):
    # Topic-level summary
    topic_summary = df.groupby('topic').agg(
        num_reviews=('review_id', 'count'),
        avg_rating=('rating', 'mean'),
        neg_review_pct=('sentiment', lambda x: (x == 'negative').mean())
    ).reset_index()

    # Constants for risk calculation
    MIN_REVIEWS_THRESHOLD = 3
    NOISE_FACTOR = 0.01
    REVENUE_NOISE_FACTOR = 0.5

    np.random.seed(42)
    
    # Calculate churn rate with minimum review threshold
    topic_summary['churn_rate'] = np.where(
        topic_summary['num_reviews'] >= MIN_REVIEWS_THRESHOLD,
        topic_summary['neg_review_pct'] * 0.5 + np.random.normal(0, NOISE_FACTOR, len(topic_summary)),
        0  # Set churn rate to 0 for topics with too few reviews
    )
    
    # Ensure churn rate is between 0 and 1
    topic_summary['churn_rate'] = topic_summary['churn_rate'].clip(0, 1)
    
    # Calculate revenue risk score with review volume weight
    topic_summary['estimated_revenue_risk_score'] = (
        topic_summary['churn_rate'] * 
        topic_summary['num_reviews'] * 
        (1 + np.random.normal(0, REVENUE_NOISE_FACTOR, len(topic_summary)).clip(-0.5, 0.5))  # Clipped noise
    )

    # Top issue-level summary
    top_issue_summary_all = df.groupby(['topic', 'top_issue']).agg(
        num_reviews=('review_id', 'count'),
        avg_rating=('rating', 'mean'),
        neg_review_pct=('sentiment', lambda x: (x == 'negative').mean())
    ).reset_index()

    # Apply same risk calculation logic to top issues
    top_issue_summary_all['churn_rate'] = np.where(
        top_issue_summary_all['num_reviews'] >= MIN_REVIEWS_THRESHOLD,
        top_issue_summary_all['neg_review_pct'] * 0.5 + np.random.normal(0, NOISE_FACTOR, len(top_issue_summary_all)),
        0
    )
    top_issue_summary_all['churn_rate'] = top_issue_summary_all['churn_rate'].clip(0, 1)
    
    top_issue_summary_all['estimated_revenue_risk_score'] = (
        top_issue_summary_all['churn_rate'] * 
        top_issue_summary_all['num_reviews'] * 
        (1 + np.random.normal(0, REVENUE_NOISE_FACTOR, len(top_issue_summary_all)).clip(-0.5, 0.5))
    )

    st.session_state['topic_summary'] = topic_summary
    st.session_state['top_issue_summary_all'] = top_issue_summary_all
    st.success("✅ Proxy metrics created successfully!")

    st.subheader("Topic-level Revenue Risk Scores")
    st.markdown("Below are the risk scores for each topic, sorted by potential revenue impact. Topics with fewer than 3 reviews are excluded for reliability.")
    
    # Show only topics with sufficient reviews
    reliable_topics = topic_summary[topic_summary['num_reviews'] >= MIN_REVIEWS_THRESHOLD]
    st.dataframe(reliable_topics.sort_values(by="estimated_revenue_risk_score", ascending=False).round(2))

    st.markdown("#### Top Issue Categories Driving Revenue Risk")
    st.markdown("""This chart highlights which **topics (issue categories)** have the **highest potential impact on revenue risk**.""")
    fig = px.bar(
        topic_summary.sort_values(by="estimated_revenue_risk_score", ascending=False),
        x="estimated_revenue_risk_score",
        y="topic",
        orientation="h",
        labels={"estimated_revenue_risk_score": "Estimated Revenue Risk Score"},
        color_discrete_sequence=["#1f77b4"],
        text="estimated_revenue_risk_score"
    )
    fig.update_layout(
        font=dict(size=14),
        yaxis_title=None,
        xaxis_title="Estimated Revenue Risk Score",
        xaxis_title_font=dict(size=16),
        plot_bgcolor="white",
        margin=dict(t=60, l=10, r=10, b=40)
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    sorted_df = topic_summary.sort_values(by="estimated_revenue_risk_score", ascending=False)
    top1, top2, top3 = sorted_df.iloc[0], sorted_df.iloc[1], sorted_df.iloc[2]
    lowest = sorted_df.iloc[-1]

    summary_text = f"""
As a product team, our key question is:
**"Which issue categories should we prioritize to improve user retention and minimize revenue risk?"** 
- For example, **{top1['topic']}** (score: {top1['estimated_revenue_risk_score']:.1f}) stands out as a top driver, indicating that unresolved issues in this area could lead to significant revenue loss or churn.
- **{top2['topic']}** and **{top3['topic']}** also rank as high-priority categories for product improvements, as they are strongly linked to user dissatisfaction and churn risk.
- Product and CX teams should investigate these topics to **identify root causes and prioritize fixes or UX improvements** accordingly.
"""
    st.markdown(summary_text)

st.markdown("---")

# Step 3: Top Issue-Level Risk Prioritization
st.subheader("Step 3. Rank Top Issues within Each Topic")

st.markdown("""
Based on the **topic-level results above**, select a topic below to **drill down** and identify which **specific issues** within that category contribute most to predicted revenue risk.
""")

all_topics = sorted(df['topic'].dropna().unique().tolist())
# Move "driver_related" to the beginning of the list if it exists
if "driver_related" in all_topics:
    all_topics.remove("driver_related")
    all_topics.insert(0, "driver_related")

selected_topic = st.selectbox("⬇️ Pick a topic to drill down", all_topics)

# Check if proxy metrics have been created
if st.session_state['top_issue_summary_all'].empty:
    st.warning("Please click 'Create Proxy Metrics' first!")
    st.stop()

# Get filtered data
top_issue_summary_all = st.session_state['top_issue_summary_all']
topic_df = top_issue_summary_all[top_issue_summary_all['topic'] == selected_topic]

# Ranking top issues by estimated revenue risk score
topic_df_sorted = topic_df.sort_values(by='estimated_revenue_risk_score', ascending=False)
topic_df_sorted = topic_df_sorted[topic_df_sorted['estimated_revenue_risk_score'] > 0]  # Filter out negative values

st.markdown(f"#### 🔍 Drill-down View: Specific Issues within `{selected_topic}`")

st.markdown("""
This chart ranks **top user complaints** within this topic by their **projected impact on revenue risk**, guiding prioritization for product teams.

> Note: We only show issues with positive revenue risk scores, as negative scores typically result from:
> - Very small number of reviews making the calculation less reliable
> - Statistical noise in the risk estimation process
> - Issues with extremely low negative review percentages
""")

if len(topic_df_sorted) == 0:
    st.info("No significant revenue risk issues found in this topic. This might be due to:\n- Low number of negative reviews\n- Very low churn risk in this category")
else:
    fig = px.bar(
        topic_df_sorted,
        x="estimated_revenue_risk_score",
        y="top_issue",
        orientation="h",
        labels={"estimated_revenue_risk_score": "Estimated Revenue Risk Score"},
        color_discrete_sequence=["#d62728"],
        text="estimated_revenue_risk_score"
    )
    fig.update_layout(
        font=dict(size=14),
        yaxis_title=None,
        xaxis_title="Estimated Revenue Risk Score",
        xaxis_title_font=dict(size=16),
        plot_bgcolor="white",
        margin=dict(t=60, l=10, r=10, b=40)
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Top issue summary text
    top_issue_name = topic_df_sorted.iloc[0]['top_issue'] if not topic_df_sorted.empty else "N/A"

    st.markdown(f"""
Within the topic **{selected_topic}**, this chart ranks the **specific user complaints** that contribute most to **estimated revenue risk**.

- The issue **{top_issue_name}** ranks highest, indicating that recurring frustrations with this aspect of our ride service may significantly impact **user retention** and **booking decisions**.
- Addressing this issue could have a direct impact on **reducing churn** and **enhancing the ride experience**.
- Product and CX teams should prioritize this finding in **roadmap planning** and consider **targeted service improvements**.
""")

st.markdown("---")

# Step 4: AI-Generated Insight Report
import base64
import io
from fpdf import FPDF

st.subheader("Step 4. AI-Generated Insight Report")

st.markdown("""
Using GPT-based analysis, we summarize business impact and product recommendations for the **top 3 issue categories** and their **top-ranked issues**.
""")

# Basic info
company = "HopSkipDrive"
industry = "ride-sharing"

topic_summary = st.session_state['topic_summary']
top_issue_summary_all = st.session_state['top_issue_summary_all']
df_total_reviews = len(df)

# Get top 3 topics (excluding 'other')
top3_topics = topic_summary[~topic_summary['topic'].str.lower().str.contains("other")].sort_values(by="estimated_revenue_risk_score", ascending=False).head(3)['topic'].tolist()

def generate_insight_openai(client, company, industry, topic, top_issue, avg_rating, num_reviews, topic_pct):
    prompt = f"""
You are a product analyst at a {industry} company called {company}.
Your job is to turn app review data into actionable insights.

Currently, the issue category '{topic}' is a top driver of estimated revenue risk.
- This topic appears in {num_reviews} user reviews, accounting for approximately {topic_pct:.1f}% of all app reviews.
- The most critical issue under this category is: "{top_issue}".
- The average rating for this issue is {avg_rating:.1f}/5.

Please provide:
1. A short explanation of why this issue may be affecting user satisfaction or revenue.
2. A clear, professional recommendation for what the product or CX team should do.

Keep your response under 4 sentences. Be insightful but concise.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior product analyst writing insights for an executive team."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating insight:\n\n{str(e)}"

insight_blocks = []

# For each top topic, select the top issue by revenue risk score and generate GPT analysis
for topic in top3_topics:
    topic_issues = top_issue_summary_all[top_issue_summary_all['topic'] == topic]
    if topic_issues.empty:
        continue

    topic_issues_sorted = topic_issues.sort_values(by='estimated_revenue_risk_score', ascending=False)
    top_issue_row = topic_issues_sorted.iloc[0]
    top_issue_name = top_issue_row['top_issue']
    avg_rating = top_issue_row['avg_rating']
    num_reviews = topic_issues['num_reviews'].sum()
    topic_pct = (num_reviews / df_total_reviews) * 100

    insight_text = generate_insight_openai(
        client, company, industry, topic, top_issue_name, avg_rating, num_reviews, topic_pct
    )

    block = f"{topic} – Top Issue: *{top_issue_name}*\n\n{insight_text}"
    insight_blocks.append(block)

    st.markdown(block)
st.markdown("---")

# Report Download Section
st.markdown("### Download Your Reports")

def create_pdf_report(insight_blocks, filename="insight_report.pdf"):
    def remove_unicode(text):
        return text.encode('ascii', 'ignore').decode('ascii')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for block in insight_blocks:
        clean_block = remove_unicode(block)
        for line in clean_block.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.ln(4)

    pdf_output = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_output).decode()

    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF Report</a>'
    return href

st.markdown(create_pdf_report(insight_blocks), unsafe_allow_html=True)

st.markdown("""A concise, GPT-generated summary of the top 3 high-risk issue categories and their most impactful sub-issues—ideal for executive briefings or product review meetings.
""")

# CSV Work Report Download
csv_buffer = io.StringIO()
combined_df = pd.concat([
    topic_summary.assign(level="Topic"),
    top_issue_summary_all.assign(level="Top Issue")
])
combined_df.to_csv(csv_buffer, index=False)
b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode()
csv_link = f'<a href="data:file/csv;base64,{b64_csv}" download="review_risk_work_report.csv">📑 Download Work Report (CSV)</a>'
st.markdown(csv_link, unsafe_allow_html=True)

st.markdown("""A full data export of topic- and issue-level metrics, including review volume, average ratings, churn rate, and estimated revenue risk scores—useful for deeper analysis or documentation.
""")
