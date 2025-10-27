import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
# --- NO ngrok IMPORTS HERE ---

# --- 1. Initialize The App ---
app = Flask(__name__)
CORS(app)  # This enables Cross-Origin Resource Sharing

# --- 2. Load All Models and Scalers at Startup ---
print("Loading all models and scalers...")
MODELS = {}
SCALERS = {}
CATEGORIES = ['large', 'mid', 'small', 'psu', 'gold']
for category in CATEGORIES:
    try:
        MODELS[category] = joblib.load(f'{category}_cap_predictor.joblib')
        SCALERS[category] = joblib.load(f'{category}_scaler.joblib')
    except FileNotFoundError:
        print(f"WARNING: Model/Scaler for {category} not found.")
print("All models loaded.")

# --- 3. Helper Function to Get Latest Data ---
def get_prediction_data(category):
    filename = f'{category}_cap_with_features.csv'
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.set_index('Date')
    
    # Re-create features
    df['lag_return_30d'] = df['NAV'].pct_change(periods=30) * 100 
    df['lag_return_90d'] = df['NAV'].pct_change(periods=90) * 100 
    df.dropna(inplace=True)

    # Define the features this model expects
    cols_to_drop = [
        'Future_Return', 'Fund_Name', 'Return', 'Month', 'Year', 'Day', 
        'NAV', 'rolling_mean_30d'
    ]
    if 'Net_FII_RsCrore' not in df.columns: # Handle Gold
        cols_to_drop.append('Net_FII_RsCrore')
        
    features = [col for col in df.columns if col not in cols_to_drop]
    
    # Get the last row of features
    latest_row_unscaled = df[features].tail(1)
    return latest_row_unscaled

# --- 4. The Main Prediction API Route ---
@app.route('/predict', methods=['GET'])
def predict():
    category = request.args.get('category')
    if category not in MODELS:
        return jsonify({'error': f"Invalid category. Choose from {CATEGORIES}"}), 400

    model = MODELS[category]
    scaler = SCALERS[category]
    latest_data = get_prediction_data(category)
    scaled_data = scaler.transform(latest_data)
    prediction = model.predict(scaled_data)
    
    return jsonify({
        'category': category,
        'predicted_return_6m': round(prediction[0], 2)
    })

# --- 5. Run the App ---
if __name__ == '__main__':
    # --- NO ngrok CODE HERE ---
    app.run(debug=True, port=5000)
    