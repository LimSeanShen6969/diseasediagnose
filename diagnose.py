import streamlit as st
import pandas as pd
import openai

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

def generate_gpt_response(question, model):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I encountered an error while processing your request."

def main():
    st.title("Smart FAQ Assistant")

    # Load data (for any potential data-related functionality you might add)
    df = load_data()

    # Define the model
    model = 'gpt-3.5-turbo'

    # User input
    user_question = st.text_input("Ask your question:")

    if st.button("Submit"):
        if user_question:
            answer = generate_gpt_response(user_question, model)
            st.write(answer)

if __name__ == "__main__":
    main()
