import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# 1. Define the EXACT preprocessor class from training
class TaxiPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.landmarks = {
            'jfk': (-73.7781, 40.6413),
            'lga': (-73.8740, 40.7769),
            'ewr': (-74.1745, 40.6895),
            'met': (-73.9632, 40.7794),
            'wtc': (-74.0099, 40.7126)
        }
        self.input_cols = [
            'passenger_count', 'pickup_datetime_month', 'pickup_datetime_weekday',
            'pickup_datetime_day', 'pickup_datetime_hour', 'trip_distance',
            'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
            'met_drop_distance', 'wtc_drop_distance', 'pickup_datetime_year',
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
        ]

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        return 6367 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def _add_features(self, df):
        df = df.copy()
        # Trip distance calculation
        df['trip_distance'] = self.haversine_np(
            df['pickup_longitude'], df['pickup_latitude'],
            df['dropoff_longitude'], df['dropoff_latitude']
        )
        
        # Datetime features
        col = 'pickup_datetime'
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_weekday'] = df[col].dt.weekday
        df[col + '_hour'] = df[col].dt.hour
        
        # Landmark distances
        for name, coords in self.landmarks.items():
            lon, lat = coords
            df[f'{name}_drop_distance'] = self.haversine_np(
                lon, lat,
                df['dropoff_longitude'],
                df['dropoff_latitude']
            )
        return df

    def transform(self, df):
        df = self._add_features(df)
        return df[self.input_cols]

    def fit(self, df, y=None):
        return self

# 2. Model loading with enhanced error handling
try:
    import sklearn
    sklearn_version = sklearn.__version__
    if sklearn_version != '1.2.2':
        print(f"WARNING: scikit-learn version mismatch (1.2.2 needed, found {sklearn_version})")
    
    model_path = os.path.join(os.path.dirname(__file__), 'taxi_fare_pipeline.joblib')
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model_artifacts = joblib.load(model_path)
    pipeline = model_artifacts['pipeline']
    
    # Access Dask model components
    model_wrapper = pipeline.named_steps['model']
    dask_model_dict = model_wrapper._Booster  # Get Dask model dictionary
    booster = dask_model_dict['booster']      # Extract actual booster
    
    # Configure for CPU prediction
    booster.set_param({
        'device': 'cpu',
        'tree_method': 'hist',
        'predictor': 'cpu_predictor'
    })
    if not booster.feature_names:
        booster.feature_names = pipeline.named_steps['preprocessor'].input_cols
        
    print(f"Model loaded! Features: {booster.feature_names}")

except Exception as e:
    print(f"Model loading failed: {str(e)}")
    booster = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not booster:
        return jsonify({
            'success': False,
            'error': 'Model not loaded - check server logs'
        }), 500

    try:
        data = request.get_json()
        required_fields = [
            'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'passenger_count'
        ]

        # Validate input
        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            return jsonify({
                'success': False,
                'error': f'Missing fields: {missing}'
            }), 400

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'pickup_datetime': data['pickup_datetime'],
            'pickup_longitude': float(data['pickup_longitude']),
            'pickup_latitude': float(data['pickup_latitude']),
            'dropoff_longitude': float(data['dropoff_longitude']),
            'dropoff_latitude': float(data['dropoff_latitude']),
            'passenger_count': int(data['passenger_count'])
        }])

        # Process datetime
        input_df['pickup_datetime'] = pd.to_datetime(input_df['pickup_datetime'])

        # Generate features
        processed_data = pipeline.named_steps['preprocessor'].transform(input_df)
        trip_distance = processed_data['trip_distance'].iloc[0]

        # Create DMatrix with proper feature names
        dtest = xgb.DMatrix(
        processed_data.values,
        feature_names=booster.feature_names,
        feature_types=['float'] * len(booster.feature_names)  # Only use 'float' or 'int'
    ,   )
    
        fare_prediction = booster.predict(dtest)[0]

        return jsonify({
            'success': True,
            'prediction': round(float(fare_prediction), 2),
            'trip_distance': round(trip_distance, 2)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)