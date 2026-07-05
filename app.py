import streamlit as st
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained SVM pipeline
model = joblib.load("svm_news_classifier.pkl")

# Load BBC dataset
df = pd.read_csv("bbc_news_dataset.csv")

st.title("📰 BBC News Classification & Recommendation System")


news = st.text_area("Enter a news article to predict its category and get similar news recommendations.")
st.markdown("### 📂 Available Categories")

st.markdown("""
<div style="display:flex; justify-content:space-between; gap:10px; margin-bottom:20px;">
    <div style="background:#1f77b4; color:white; padding:10px 18px; border-radius:10px; font-weight:bold; text-align:center; flex:1;">
        💼 Business
    </div>
    <div style="background:#e83e8c; color:white; padding:10px 18px; border-radius:10px; font-weight:bold; text-align:center; flex:1;">
        🎭 Entertainment
    </div>
    <div style="background:#fd7e14; color:white; padding:10px 18px; border-radius:10px; font-weight:bold; text-align:center; flex:1;">
        🏛 Politics
    </div>
    <div style="background:#28a745; color:white; padding:10px 18px; border-radius:10px; font-weight:bold; text-align:center; flex:1;">
        ⚽ Sport
    </div>
    <div style="background:#6f42c1; color:white; padding:10px 18px; border-radius:10px; font-weight:bold; text-align:center; flex:1;">
        💻 Tech
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.stButton > button {
    background-color: #00C853;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    width: 100%;
    border: none;
}

div.stButton > button:hover {
    background-color: #00E676;
    color: black;
}
</style>
""", unsafe_allow_html=True)
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        # Predict category
        prediction = model.predict([news])[0]

        st.success(f"Predicted Category: {prediction}")


        # Filter same category
        category_articles = df[df["Category"] == prediction]

        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(category_articles["Text"])

        new_vector = vectorizer.transform([news])

        similarity = cosine_similarity(new_vector, tfidf_matrix)

        top3 = similarity.argsort()[0][-3:][::-1]

        st.subheader("Top 3 Similar Articles")

        for i, idx in enumerate(top3, start=1):

            st.markdown(f"### {i}")

            st.write(category_articles.iloc[idx]["Text"])

            st.divider()
