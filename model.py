import pickle
import numpy as np

# Load the trained model
def load_model():
    with open("wine_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Define the prediction function
def predict_wine_quality(data):
    model = load_model()
    data = np.array(data).reshape(1, -1)  
    prediction = model.predict(data)
    return prediction.tolist()

if __name__ == "__main__":
    print("Enter the wine features:")
    
    try:
        # Take user input for each feature
        fixed_acidity = float(input("Fixed Acidity: "))
        volatile_acidity = float(input("Volatile Acidity: "))
        citric_acid = float(input("Citric Acid: "))
        residual_sugar = float(input("Residual Sugar: "))
        chlorides = float(input("Chlorides: "))
        free_sulfur_dioxide = float(input("Free Sulfur Dioxide: "))
        density = float(input("Density: "))
        pH = float(input("pH: "))
        sulphates = float(input("Sulphates: "))
        alcohol = float(input("Alcohol: "))

        # Create input data list
        sample_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                        free_sulfur_dioxide, density, pH, sulphates, alcohol]]
        
        # Predict quality
        print("Predicted Quality:", predict_wine_quality(sample_data))

    except ValueError:
        print("Invalid input! Please enter numeric values for all fields.")

""" 
sample_data_list = [
        [7.0, 0.27, 0.36, 20.7, 0.045, 45, 1.001, 3.00, 0.45, 8.8],
        [6.3, 0.30, 0.34, 1.6, 0.049, 14, 0.992, 3.30, 0.49, 9.5],
        [8.1, 0.28, 0.40, 6.9, 0.050, 30, 0.995, 3.26, 0.44, 10.1],
        [5.8, 0.29, 0.32, 2.0, 0.047, 22, 0.993, 3.38, 0.46, 9.2],
        [7.2, 0.25, 0.35, 8.5, 0.045, 55, 0.998, 3.15, 0.42, 10.5],
    ]
"""