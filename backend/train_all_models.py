import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib 

FILES_TO_TRAIN = {
    'large': 'large_cap_with_features.csv',
    'mid': 'mid_cap_with_features.csv',
    'small': 'small_cap_with_features.csv',
    'psu': 'psu_cap_with_features.csv',
    'gold': 'gold_cap_with_features.csv'
}

print("--- Starting to train all 5 models ---")

for category, filename in FILES_TO_TRAIN.items():
    print(f"\n🤖 Training model for: {category.upper()} CAP")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(filename, parse_dates=['Date'])
        df = df.set_index('Date')
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{filename}'. Skipping.")
        continue

    # --- 2. Feature Engineering ---
    prediction_horizon = 180
    df['Future_Return'] = df['NAV'].pct_change(periods=prediction_horizon).shift(-prediction_horizon) * 100
    df['Future_Return'] = df['Future_Return'].clip(lower=-75, upper=100)
    
    df['lag_return_30d'] = df['NAV'].pct_change(periods=30) * 100 
    df['lag_return_90d'] = df['NAV'].pct_change(periods=90) * 100 
    df.dropna(inplace=True)

    # --- 3. Prepare Data for XGBoost ---
    cols_to_drop = [
        'Future_Return', 'Fund_Name', 'Return', 'Month', 'Year', 'Day', 
        'NAV', 'rolling_mean_30d' # Remove non-signal features
    ]
    # Handle gold file not having FII
    if 'Net_FII_RsCrore' not in df.columns:
        cols_to_drop.append('Net_FII_RsCrore') 
        
    features = [col for col in df.columns if col not in cols_to_drop]
    X = df[features]
    y = df['Future_Return']

    # --- 4. Split and Scale ---
    split_index = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)

    # --- 5. Train Model ---
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        objective='reg:squarederror', random_state=42, n_jobs=-1
    )
    xgb_regressor.fit(X_train, y_train, verbose=False)
    
    # --- 6. Save Model AND Scaler ---
    model_filename = f'{category}_cap_predictor.joblib'
    scaler_filename = f'{category}_scaler.joblib' # The new, critical file
    
    joblib.dump(xgb_regressor, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    # --- 7. Evaluate and Print ---
    y_pred = xgb_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Model complete for {category.upper()}! (RMSE: {rmse:.2f}%)")
    print(f"💾 Model saved as: '{model_filename}'")
    print(f"💾 Scaler saved as: '{scaler_filename}'")

print("\n--- 🎉 All models and scalers have been trained and saved! ---")