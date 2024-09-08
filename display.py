from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your trained ARIMA model
try:
    model = joblib.load('forcasting.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/api/forecast', methods=['POST'])
def forecast():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        print(f"Received data: {data}")

        input_number = data.get('number')

        if input_number is None:
            return jsonify({"error": "Input number is missing"}), 400

        try:
            input_number = float(input_number)
            print(f"Input number (validated): {input_number}")
        except ValueError:
            return jsonify({"error": "Invalid number format"}), 400

        forecast_steps = int(input_number)  # Use the input number for forecast steps
        future_forecast = model.forecast(steps=forecast_steps)

        time_index = pd.date_range(start='2024-08-27 00:00', periods=forecast_steps, freq='H')

        forecast_df = pd.DataFrame({'time': time_index, 'forecasted_demand': future_forecast})

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set(style="whitegrid")
        sns.lineplot(x='time', y='forecasted_demand', data=forecast_df, color='#2f7ed8', linewidth=3, ax=ax)

        max_value = forecast_df['forecasted_demand'].max()
        min_value = forecast_df['forecasted_demand'].min()
        max_time = forecast_df.loc[forecast_df['forecasted_demand'] == max_value, 'time'].values[0]
        min_time = forecast_df.loc[forecast_df['forecasted_demand'] == min_value, 'time'].values[0]

        ax.annotate(f"{max_value:.2f}", xy=(max_time, max_value), xytext=(max_time, max_value + (max_value * 0.05)),
                    arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle="->", lw=1.5),
                    fontsize=10, color='green')

        ax.annotate(f"{min_value:.2f}", xy=(min_time, min_value), xytext=(min_time, min_value - (min_value * 0.05)),
                    arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=1.5),
                    fontsize=10, color='red')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.title('Bike Sharing System - Forecasted Demand of next ' + str(int(input_number)) + ' hours', fontsize=16, weight='bold', color='#333333')
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.ylabel('Forecasted Demand', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close(fig)

        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/heatmap', methods=['GET'])
def heatmap():
    try:
        # Load the data
        df = pd.read_csv('trip.csv', on_bad_lines='skip')

        # Count the number of trips starting from each station
        station_counts = df['from_station_name'].value_counts().head(10)

        # Create a DataFrame for the heatmap data
        station_counts_df = pd.DataFrame(station_counts).T

        # Create a plot and save it to a BytesIO object
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.set(style="whitegrid")
        sns.heatmap(station_counts_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

        ax.set_title('Top 10 Most Used Stations (Number of Trips)', fontsize=16, weight='bold', color='#333333')
        ax.set_xlabel('Station Name')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close(fig)

        return send_file(img, mimetype='image/png')

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
