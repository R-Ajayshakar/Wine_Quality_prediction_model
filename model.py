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
    sample_data = [[7,0.27,0.36,20.7,0.045,45,1.001,3,0.45,8.8]]
    print("Predicted Quality:", predict_wine_quality(sample_data))
