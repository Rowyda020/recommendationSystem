import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Load the DataFrame and pre-process
df1 = pd.read_csv('new.csv', engine='python')
df1 = df1.dropna()

# Initialize TF-IDF vectorizer
tdif = TfidfVectorizer(stop_words='english')

# Fill missing values in jobdescription column
df1['jobdescription'] = df1['jobdescription'].fillna('')

# Compute TF-IDF matrix
tdif_matrix = tdif.fit_transform(df1['jobdescription'])

# Compute cosine similarity matrix
cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)

# Create index for job titles
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]
    tech_indices = [i[0] for i in sim_scores]
    return df1['jobtitle'].iloc[tech_indices]

# Streamlit UI
st.header('Tech Jobs Recommender')

toon_list = df1['jobtitle'].values
selected_toon = st.selectbox(
    "Type or select a job from the dropdown",
    toon_list
)

if st.button('Show Recommendation'):
    recommended_toon_names = get_recommendations(selected_toon)
    for i in recommended_toon_names:
        st.subheader(i)
