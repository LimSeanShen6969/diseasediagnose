import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score

# Function to load your dataset
def load_data():
    # Replace 'your_dataset.csv' with the actual filename
    try:
        data = pd.read_csv('qa_dataset_with_embeddings.csv') 
        return data
    except pd.errors.ParserError as e:
        st.error(f"Error reading CSV file: {e}")
        # Print more context about the error
        st.write(f"Error details: {e.args}")
        return None

# Function to preprocess the data (if needed)
def preprocess_data(data):
    # Handle missing values, feature scaling, etc.
    data['Question'] = data['Question'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)
    return data

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Streamlit app
def main():
    st.title("Classification Demo")

    # Load and preprocess data
    data = load_data()
    if data is None: # Stop execution if there was an error loading the data
        return
    
    data = preprocess_data(data)

    # Split data into features (X) and target (y)
    X = data.drop('target_column', axis=1)  # Replace 'target_column' 
    y = data['target_column']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Correct indentation

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    

    st.write(f"Accuracy: {accuracy}")

    # User input for prediction
    st.subheader("Predict New Data")
    new_data = st.text_input("Enter feature values (comma-separated)")

    if st.button("Predict"):
        try:
            input_values = [float(x) for x in new_data.split(",")]
            prediction = model.predict([input_values])
            st.write(f"Prediction: {prediction[0]}")
        except ValueError:
            st.error("Invalid input. Please enter comma-separated numeric values.")

if __name__ == "__main__":
    main()
