import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Replace with your chosen embedding model
from sentence_transformers import SentenceTransformer

def load_data():
    try:
        df = pd.read_csv("qa_dataset_with_embeddings.csv")
        return df
    except FileNotFoundError:
        st.error("File not found. Please make sure 'qa_dataset_with_embeddings.csv' exists in the correct location.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please check the file format.")
        st.stop()

def load_embeddings(data):
    if data is not None:
        embeddings = data['Question_Embedding'].values
        embeddings = np.array(list(map(eval, embeddings)))
        return embeddings
    else:
        return None

# Generate embedding for a new question
def generate_embedding(question, model):
    embedding = model.encode(question)
    return embedding

# Find the most similar answer
def find_answer(question_embedding, embeddings, data, threshold=0.7):
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]

    if similarity_score >= threshold:
        answer = data['Answer'][most_similar_index]
        return answer, similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None

# Streamlit app
def main():
    st.title("Smart FAQ Assistant")

    # Load data and embeddings
    df = load_data()
    embeddings = load_embeddings(df)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # User input
    user_question = st.text_input("Ask your question:")

    if st.button("Submit"):
        if user_question:
            question_embedding = generate_embedding(user_question, model)
            answer, similarity = find_answer(question_embedding, embeddings, df)
            st.write(answer)
            if similarity:
                st.write(f"Similarity Score: {similarity:.2f}")

if __name__ == "__main__":
    main()
