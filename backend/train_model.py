import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib 
from sklearn.preprocessing import StandardScaler

# --- 0. Verify Version ---
print(f"✅ Script is using XGBoost version: {xgb.__version__}")

# --- 1. Load the Prepared Data ---
file_to_train = 'large_cap_with_features.csv' 
try:
    df = pd.read_csv(file_to_train, parse_dates=['Date'])
    df = df.set_index('Date')
    print(f"✅ Successfully loaded '{file_to_train}'")
except FileNotFoundError:
    print(f"❌ Error: '{file_to_train}' not found. Make sure it's in the same folder.")
    exit()

# --- 2. Feature Engineering ---
print("⚙️  Starting feature engineering...")

prediction_horizon = 180
df['Future_Return'] = df['NAV'].pct_change(periods=prediction_horizon).shift(-prediction_horizon) * 100
df['Future_Return'] = df['Future_Return'].clip(lower=-75, upper=100)
print(f"✅ Outliers clipped. Max future return is now: {df['Future_Return'].max():.2f}")

# --- We are now using *only* stationary features (returns) ---
df['lag_return_30d'] = df['NAV'].pct_change(periods=30) * 100 
df['lag_return_90d'] = df['NAV'].pct_change(periods=90) * 100 
# We remove rolling_mean_30d as it's a non-stationary price level
df.dropna(inplace=True)

# --- 3. Prepare Data for XGBoost ---

# --- THIS IS THE KEY FIX ---
# We are removing NAV and rolling_mean_30d.
# The model will now learn from signals, not from price.
# ------------------------------
cols_to_drop = [
    'Future_Return', 'Fund_Name', 'Return', 'Month', 'Year', 'Day', 
    'NAV', # This is the price level, not a signal
    'rolling_mean_30d' # This is also a price level
]
features = [col for col in df.columns if col not in cols_to_drop]
# ------------------------------

X = df[features]
y = df['Future_Return']

print("\n✅ Features being sent to the model (Signals Only):")
print(X.info()) # This will now show only our signals

# --- 4. Split and Scale Data ---
split_index = int(len(X) * 0.8)
X_train_unscaled, X_test_unscaled = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("⚖️  Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)
print("✅ Features scaled successfully.")

print(f"\n📊 Data split into training and testing sets:")
print(f"Training set size: {len(X_train)} days")
print(f"Testing set size: {len(X_test)} days")

# --- 5. Calculate Baseline Error ---
# This is the error if we just guess the average every time
baseline_pred = np.full(shape=y_test.shape, fill_value=y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"\n📈 Baseline RMSE (Just guessing average): {baseline_rmse:.2f}%")
print("(Our model's RMSE must be lower than this to be useful)")

# --- 6. Train the XGBoost Model ---
print("\n🤖 Training the XGBoost model on signals...")

xgb_regressor = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

xgb_regressor.fit(X_train, y_train, verbose=False)

print("✅ Model training complete!")

# --- 7. Save the Trained Model ---
model_filename = 'large_cap_predictor.joblib'
joblib.dump(xgb_regressor, model_filename)
joblib.dump(scaler, 'scaler.joblib')
print(f"💾 Model saved as '{model_filename}'")
print(f"💾 Scaler saved as 'scaler.joblib'")

# --- 8. Evaluate the Model ---
print("\n📈 Evaluating REAL model performance...")
y_pred = xgb_regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}%")

if rmse < baseline_rmse:
    print("✅ SUCCESS: The model is smarter than just guessing the average!")
else:
    print("❌ WARNING: The model is not performing better than a simple average.")

# --- 9. Plotting the Results ---
print("\nGenerating prediction plot...")
results_df = pd.DataFrame({'Actual_Return': y_test, 'Predicted_Return': y_pred}, index=y_test.index)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))
plt.plot(results_df['Actual_Return'], label='Actual 6-Month Return', color='royalblue', linewidth=2)
plt.plot(results_df['Predicted_Return'], label='Predicted 6-Month Return', color='red', linestyle='--', alpha=0.8)
plt.title('Large Cap Fund: Actual vs. Predicted 6-Month Returns', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Return (%)', fontsize=12)
plt.legend()
plt.show()

print("\n🎉 All done! Close the plot window to finish.")