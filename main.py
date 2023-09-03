import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the trained model
def load_model(model_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the scaler (normalizer)
def load_scaler(scaler_filename):
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Function to preprocess user input and make predictions
def predict_pulsar(model, scaler, input_data):
    # Preprocess the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    # Predict whether the record is a pulsar or not
    prediction = model.predict(input_data_scaled)
    return prediction

if __name__ == "__main__":
    # Load the trained model and scaler
    scaler_filename = "scaler.pkl"
    model_filename = "pulsar_model.pkl"
    trained_model = load_model(model_filename)
    loaded_scaler = load_scaler(scaler_filename)
    # Get user input for feature values
    input_data = []
    for i in range(8):
        value = float(input(f"Enter value for feature {i+1}: "))
        input_data.append(value)

    # Make a prediction using the loaded model and scaler
    prediction = predict_pulsar(trained_model, loaded_scaler, np.array(input_data))

    # Display the prediction result
    print(f"Prediction: {'Pulsar' if prediction == 1 else 'Not Pulsar'}")
