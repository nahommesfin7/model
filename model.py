from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from changer import DynamicPricing
import pandas as pd
import csv
import os

# Initialize Flask app
app = Flask(__name__)

# Allow CORS for all domains on all routes
CORS(app)

# Load your model with error handling
try:
    model = tf.keras.models.load_model('dynamicpricing.h5')
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Function to store data in a CSV file
def store_pricing_data_to_csv(start_time, duration, output_value, is_peak_time):
    file_exists = os.path.isfile('pricing_data.csv')
    try:
        with open('pricing_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty to write headers
            if not file_exists:
                writer.writerow(['Start Time', 'Duration', 'Output Value', 'Is Peak Time'])
            writer.writerow([start_time, duration, output_value, is_peak_time])
        print("Data stored in CSV successfully.")
    except Exception as e:
        print(f"Error storing data to CSV: {str(e)}")

# Define your API route
@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        # Get data from the request body
        data = request.json
        print(f"Received data: {data}")
        
        # Extract startTime and duration
        start_time_str = data.get('startTime')
        duration = data.get('duration')

        # Handle case where duration is less than 5 minutes
        if duration is None:
            return jsonify({"error": "Duration is missing"}), 400
        
        if duration < 5:
            return jsonify({"output": 'Your ride is free'}) 

        # Ensure duration is a valid integer
        try:
            duration = int(duration)
            print(f"Duration (validated): {duration}")
        except ValueError:
            return jsonify({"error": "Invalid duration format"}), 400
        
        # Convert start time to pandas datetime
        try:
            start_time = pd.to_datetime(start_time_str)
            print(f"Start time (validated): {start_time}")
        except Exception as e:
            return jsonify({"error": "Invalid startTime format"}), 400

        # Load training data for scaling
        training_data = pd.read_csv('Bike_Share_Trip_Data.csv')
        
        # Fit the scaler using training data (if required)
        try:
            model.fit_scaler(training_data)  # Check if this step is required
        except Exception as e:
            print(f"Scaler fitting error: {str(e)}")

        # Instantiate DynamicPricing and check for peak time
        pricing = DynamicPricing()
        is_peak = pricing.is_peak_time(start_time)
        print(f"Is peak time: {is_peak}")
        
        # Prepare the input data for the model
        input_data = np.array([duration, int(is_peak)]).reshape(1, -1)
        print(f"Input data for model: {input_data}")

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        print(f"Model prediction: {prediction}")

        # Extract the prediction value and format it
        output_value = round(prediction[0].tolist()[0] / 1000, 2) 
        print(f"Output value: {output_value}")
        
        # Store the data to CSV
        store_pricing_data_to_csv(start_time_str, duration, output_value, is_peak)

        # Return the result as JSON
        return jsonify({"output": output_value, "is_peak_time": is_peak})

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
