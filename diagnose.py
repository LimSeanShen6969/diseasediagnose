import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set the OpenAI API key correctly
openai.api_key = st.secrets["mykey"]

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

def generate_embedding(question, model):
    embedding = model.encode(question)
    return embedding

def find_answer_gpt(question, api_key):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        api_key=api_key
    )
    return response.choices[0].message['content']

def find_answer(question_embedding, embeddings, data, threshold=0.7):
    if embeddings is not None and len(embeddings) > 0:
        similarities = cosine_similarity([question_embedding], embeddings)[0]
        most_similar_index = np.argmax(similarities)
        similarity_score = similarities[most_similar_index]

        if similarity_score >= threshold:
            answer = data['Answer'][most_similar_index]
            return answer, similarity_score
        else:
            return find_answer_gpt(question, openai.api_key), None
    else:
        return "No embeddings available.", None

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
