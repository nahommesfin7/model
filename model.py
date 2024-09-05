from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from changer import DynamicPricing  
import pandas as pd  

# Initialize Flask app
app = Flask(__name__)

# Allow CORS for all domains on all routes
CORS(app)

# Load your model
model = tf.keras.models.load_model('dynamicpricing.h5')

# Define your API route
@app.route('/api/process', methods=['POST'])
def process_data():
    try:
        # Get data from the request body
        data = request.json
        print(data)
        
        # Extract the required fields
        start_time_str = data.get('startTime')
        duration = data.get('duration')

        # Convert start time to a pandas datetime object
        start_time = pd.to_datetime(start_time_str)
        
        # Instantiate DynamicPricing and check for peak time
        pricing = DynamicPricing()
        is_peak = pricing.is_peak_time(start_time)
       
        # Prepare the input data for the model
        input_data = np.array([duration, int(is_peak)]).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)

        # Extract the prediction value and divide it by 100 as per your code logic
        output_value = round(prediction[0].tolist()[0],2) 
        
        # Return the result as JSON
        return jsonify({"output": output_value, "is_peak_time": is_peak})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
