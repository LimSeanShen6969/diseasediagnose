import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Set the OpenAI API key correctly
openai.api_key = st.secrets["mykey"]

# Function to load the data
def load_data():
    try:
        df = pd.read_csv("qa_dataset_with_embeddings.csv")
        st.write("Data loaded successfully!")
        st.write(df.columns)  # Print column names to debug
        return df
    except FileNotFoundError:
        st.error("File not found. Please make sure 'qa_dataset_with_embeddings.csv' exists in the correct location.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please check the file format.")
        st.stop()

# Function to load embeddings
def load_embeddings(data):
    if data is not None:
        if 'Question_Embedding' in data.columns:
            embeddings = data['Question_Embedding'].values
            embeddings = np.array(list(map(eval, embeddings)))
            return embeddings
        else:
            st.error("Column 'Question_Embedding' not found in the data.")
            st.stop()
    else:
        return None

# Function to generate embedding for a new question
def generate_embedding(question):
    response = openai.Embedding.create(input=question, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Function to find the most similar answer
def find_answer(question_embedding, embeddings, data, threshold=0.7):
    if embeddings is not None and len(embeddings) > 0:
        similarities = cosine_similarity([question_embedding], embeddings)[0]
        most_similar_index = np.argmax(similarities)
        similarity_score = similarities[most_similar_index]

        if similarity_score >= threshold:
            answer = data['Answer'][most_similar_index]
            return answer, similarity_score
        else:
            return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None
    else:
        return "No embeddings available.", None

# Streamlit app
def main():
    st.title("Smart FAQ Assistant")

    # Load data and embeddings
    df = load_data()
    embeddings = load_embeddings(df)

    # Initialize session state variables
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    if 'answer' not in st.session_state:
        st.session_state.answer = ""
    if 'similarity' not in st.session_state:
        st.session_state.similarity = None

    # User input
    st.session_state.user_question = st.text_input("Ask your question:", value=st.session_state.user_question)

    # Add "Submit" and "Clear" buttons
    if st.button("Submit"):
        if st.session_state.user_question:
            question_embedding = generate_embedding(st.session_state.user_question)
            answer, similarity = find_answer(question_embedding, embeddings, df)
            st.session_state.answer = answer
            st.session_state.similarity = similarity

    if st.button("Clear"):
        st.session_state.user_question = ""
        st.session_state.answer = ""
        st.session_state.similarity = None

    # Display answer and similarity score
    if st.session_state.answer:
        st.write(st.session_state.answer)
        if st.session_state.similarity is not None:
            st.write(f"Similarity Score: {st.session_state.similarity:.2f}")

    # Rating functionality
    if st.session_state.answer:
        rating = st.slider("Rate the helpfulness of the answer (1-5):", min_value=1, max_value=5)
        st.write(f"Your rating: {rating}")

    # Display common FAQs
    if st.checkbox("Show Common FAQs"):
        st.write(df[['Question', 'Answer']].head(10))

if __name__ == "__main__":
    main()
