# --- Streamlit App with Top Issue and Toggle Sub Issues (Improved Version) ---
import streamlit as st
import pandas as pd
import openai
import os

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("HopSkipDrive Review Intelligence Dashboard")
st.markdown("### Top Issues from Negative Reviews")
st.markdown("""
This page surfaces the **most common user-reported issues** from negative HopSkipDrive app reviews to help product and CX teams **uncover pain points that impact satisfaction and retention**.  
**Filter by topic** to dive into **high-volume complaints**, track their average ratings, and review concrete examples of what users are saying — organized by top-level and sub-level issues. **AI-generated summaries** are included to quickly grasp each issue's definition and its potential impact on product experience.
""")

st.markdown("---")

# --- OpenAI 설정 ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

client = openai.Client(api_key=OPENAI_API_KEY)

# --- Load Data ---
DEFAULT_FILE_PATH = "hopskip_subissues.csv"
uploaded_file = st.sidebar.file_uploader("📂 Upload your review data", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded file loaded successfully.")
    else:
        df = pd.read_csv(DEFAULT_FILE_PATH)
        st.sidebar.markdown(
            f"<p style='font-size: 0.8em; color: #444;'>No file uploaded. Using default file: <code>{DEFAULT_FILE_PATH}</code></p>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    st.error(f"❌ Default file '{DEFAULT_FILE_PATH}' not found.")
    st.stop()

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# --- Filter Data ---
df_filtered = df[
    (~df["sentiment"].isin(["Positive", "Neutral"])) &  # Exclude positive and neutral sentiments
    (df["rating"] <= 3) &  # Only include ratings of 3 or less
    (df["topic"].notna()) & 
    (df["aspect"].notna())
]

# --- Topic Statistics 생성 ---
topic_stats = {}
for topic in df_filtered["topic"].unique():
    topic_df = df_filtered[df_filtered["topic"] == topic]
    total_reviews = len(topic_df)
    avg_rating = round(topic_df["rating"].mean(), 1)
    review_percentage = round((total_reviews / len(df_filtered)) * 100, 1)
    topic_stats[topic] = {
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "review_percentage": review_percentage
    }

# --- Sidebar: Topic Selection ---
st.sidebar.markdown("### 🔍 Select Topic to Analyze")

def format_topic_label(topic):
    stats = topic_stats[topic]
    return f"""
    <strong>{topic.capitalize()}</strong><br>
    <span style='font-size: 0.9em;'>Review# {stats['total_reviews']} &nbsp;&nbsp; Review% {stats['review_percentage']}% &nbsp;&nbsp; Avg. Rating {stats['avg_rating']}/5</span>
    """

topics_sorted = sorted(topic_stats.keys(), key=lambda x: -topic_stats[x]["total_reviews"])
radio_html = """<style>
    .stRadio > div > label > div:first-child {
        display: flex;
        flex-direction: column;
        line-height: 1.4;
    }
</style>"""
st.sidebar.markdown(radio_html, unsafe_allow_html=True)

selected_topic = st.sidebar.radio(
    label="📱 Topics",
    options=topics_sorted,
    format_func=lambda x: x
)

# --- Topic Summary ---
st.sidebar.markdown(
    f"""
    <div style='background-color:#EDE7F6; padding:10px; border-left: 5px solid #7B1FA2; border-radius: 8px; margin-top: 10px;'>
        <strong>{selected_topic.capitalize()}</strong><br>
        <span style='font-size: 0.9em;'>Review# {topic_stats[selected_topic]['total_reviews']} &nbsp;&nbsp; Review% {topic_stats[selected_topic]['review_percentage']}% &nbsp;&nbsp; Avg. Rating {topic_stats[selected_topic]['avg_rating']}/5</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# --- Sidebar: All Topics Overview ---
st.sidebar.markdown("### 📊 All Topics Overview")
for topic in topics_sorted:
    stats = topic_stats[topic]
    st.sidebar.markdown(
        f"""
        <div style='background-color:#F3E5F5; padding:8px; border-radius:4px; margin-bottom:8px;'>
            <strong>{topic.capitalize()}</strong><br>
            <span style='font-size: 0.9em;'>Review# {stats['total_reviews']} &nbsp;&nbsp; Review% {stats['review_percentage']}% &nbsp;&nbsp; Avg. Rating {stats['avg_rating']}/5</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Section Header ---
st.subheader(f"Top {selected_topic} Issues")
issues_df = df_filtered[df_filtered["topic"] == selected_topic]

# --- OpenAI 응답 함수 ---
def get_issue_definition(issue_name, example_reviews):
    try:
        reviews_text = "\n".join(f"- {review}" for review in example_reviews[:3])
        prompt = f"""
        Based on the issue name and example reviews, provide a structured response with:

        **Issue:** {issue_name}

        **Example Reviews:**
        {reviews_text}

        Please provide:
        Two concise sentences:  
        1. A brief definition of the issue.  
        2. How this issue affects user experience and product performance.  

        Do NOT include labels such as 'Issue Definition' or 'Impact on User Experience'.  
        ONLY return two sentences without any extra text.  
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a product analyst providing clear, structured issue definitions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating definition: {str(e)}"

# --- Top Issues Loop ---
issue_review_counts = issues_df.groupby("top_issue").size().sort_values(ascending=False)
top_issues_sorted = issue_review_counts.index.tolist()

for idx, top_issue in enumerate(top_issues_sorted, start=1):
    group_df = issues_df[issues_df["top_issue"] == top_issue]
    total_reviews = len(group_df)
    avg_rating = round(group_df["rating"].mean(), 2)
    review_pct = round((total_reviews / len(issues_df)) * 100, 2)

    example_reviews = group_df["review"].dropna().sample(min(3, len(group_df))).tolist()
    issue_analysis = get_issue_definition(top_issue, example_reviews)

    st.markdown(
        f"""
        <div style="background-color:#EDE7F6; padding:18px; border-radius:8px; border-left: 6px solid #673AB7; margin-bottom:10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color:#4A148C; margin-bottom: 0;">Top Issue {idx}. {top_issue}</h4>
                <span style="background-color:#D1C4E9; padding: 6px 12px; border-radius: 6px; font-size: 14px; color: #4A148C;">
                    Review# {total_reviews} &nbsp;|&nbsp; Review% {review_pct}% &nbsp;|&nbsp; Avg. Rating {avg_rating}/5
                </span>
            </div>
            <p style="font-size:15px; color:#333;">
                {issue_analysis}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Case 1: sub_issue_index가 없는 경우 → 리뷰 모두 표시
    sub_issue_null_df = group_df[group_df["sub_issue_index"].isna()]
    if len(sub_issue_null_df) > 0:
        st.markdown("#### 💬 Lowest-Rated Review Highlighting This Issue")

        # ⭐ 리뷰 중에서 별점이 가장 낮은 것 1개 선택
        row = sub_issue_null_df.loc[sub_issue_null_df["rating"].idxmin()]
        aspect_index = row["aspect_index"] if "aspect_index" in row and pd.notna(row["aspect_index"]) else "N/A"
        rating = row["rating"] if pd.notna(row["rating"]) else "N/A"

        st.markdown(f"""
        <div style="border: 1px solid #D1C4E9; border-radius:8px; padding:12px; margin-bottom:10px; background-color:#f3ecff;">
            <p style='font-size:15px; margin-bottom:8px;'><em>{row['review']}</em></p>
            <div style='display: flex; justify-content: space-between; font-size:14px;'>
                <span>⭐ <strong>Rating:</strong> {rating}/5</span>
                <span><strong>Issue Index:</strong> {aspect_index}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Case 2: sub_issue_index가 있는 경우 → 서브 이슈 별로 sample 하나만 보여주고 나머진 index로 나열
    sub_df = group_df[group_df["sub_issue_index"].notna()]
    if len(sub_df) > 0:

        # 💄 CSS 스타일: Expander 핑크 배경
        st.markdown("""
        <style>
        [data-testid="stExpander"] > div:first-child {
            background-color: #FCE4EC;
            border-radius: 6px;
            padding: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.expander("Toggle Sub Issues"):
            unique_aspects = sub_df["aspect"].unique()
            max_sub_issues_to_show = 5
            limited_aspects = unique_aspects[:max_sub_issues_to_show]

            for i, aspect in enumerate(limited_aspects, start=1):
                sub_group = sub_df[sub_df["aspect"] == aspect].copy()
                st.markdown(f"#### Sub Issue #{i}: {aspect}")

                sample = sub_group.sample(1).iloc[0]
                aspect_index = sample["aspect_index"] if "aspect_index" in sample and pd.notna(sample["aspect_index"]) else "N/A"
                rating = sample["rating"] if pd.notna(sample["rating"]) else "N/A"

                st.markdown(f"""
                <div style="border: 1px solid #F8BBD0; border-radius:8px; padding:12px; margin-bottom:10px; background-color:#fce4ec;">
                    <p style='font-size:15px; margin-bottom:8px;'><em>{sample['review']}</em></p>
                    <div style='display: flex; justify-content: space-between; font-size:14px;'>
                        <span>⭐ <strong>Rating:</strong> {rating}/5</span>
                        <span><strong>Issue Index:</strong> {aspect_index}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                remaining_indices = sub_group[~sub_group.index.isin([sample.name])]["aspect_index"].dropna().tolist()
                if remaining_indices:
                    index_list = ", ".join(str(x) for x in remaining_indices)
                    st.markdown(f"Related indices: {index_list}")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
