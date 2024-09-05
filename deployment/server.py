from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

# DataFrame to store data and predictions
data_store = pd.DataFrame(columns=['data', 'prediction'])

# Load the machine learning model
model = pickle.load(open('/models/sample_ml_model.pkl', 'rb'))

@app.route("/predict", methods=['POST'])
def predict():
    # Get JSON data from request body
    data = request.get_json()
    
    # Assuming data is a list with a key 'array'
    prediction = model.predict([data['array']])
    
    # Store the input data and prediction
    data_store.loc[len(data_store)] = [data, prediction[0]]
    
    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

@app.route("/", methods=['GET'])
def home():
    # Display the number of past predictions and the predictions in tabular format
    num_predictions = len(data_store)
    print(num_predictions)
    
    # Return the data store in dictionary format
    return jsonify(data_store.to_dict())

if __name__ == "__main__":
    app.run(port=5000, host='127.0.0.1', debug=True)