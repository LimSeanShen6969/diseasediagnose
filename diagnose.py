import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Function to load your dataset
def load_data():
    try:
        data = pd.read_csv('qa_dataset_with_embeddings.csv')
        return data
    except pd.errors.ParserError as e:
        print(f"Error reading CSV: {e}")
        # Optionally print a few lines around row 502 for inspection
        with open('qa_dataset_with_embeddings.csv', 'r') as f:
            lines = f.readlines()
            for i in range(500, 505):  # Adjust range as needed
                if i < len(lines):
                    print(lines[i])
        return None

# Function to preprocess the data (if needed)
def preprocess_data(data):
    # Handle missing values, feature scaling, etc.
    # ...
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

    # Check if data loaded successfully
    if data is None:
        st.error("Error loading data. Please check the file path and format.")
        return # Stop execution if no data is loaded

    data = preprocess_data(data)

    # Split data into features (X) and target (y)
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual name of your target column
    y = data['target_column'] # Replace 'target_column' with the actual name of your target column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
