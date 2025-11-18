# app/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -------------------------------
# Load pipeline and label encoder
# -------------------------------
# Compute absolute path to artifacts folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline_path = os.path.join(BASE_DIR, 'artifacts', 'pipeline.joblib')

# Load the saved pipeline and label encoder
try:
    artifacts = joblib.load(pipeline_path)
    pipeline = artifacts['pipeline']
    le = artifacts['label_encoder']
except FileNotFoundError:
    raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Symptom Tracker',
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease based on symptom input"""
    try:
        # Get JSON payload from POST request
        payload = request.get_json(force=True)
        df_input = pd.DataFrame([payload])  # expects JSON with symptom columns

        # Fill missing features with 0
        feature_cols = pipeline.named_steps['model'].feature_names_in_
        for col in feature_cols:
            if col not in df_input:
                df_input[col] = 0

        # Ensure correct column order
        df_input = df_input[feature_cols]

        # Make predictions
        preds = pipeline.predict(df_input)
        proba = pipeline.predict_proba(df_input)

        # Decode label and prepare response
        disease_preds = le.inverse_transform(preds)
        results = []
        for disease, prob in zip(disease_preds, proba):
            results.append({'disease': disease, 'probability': float(max(prob))})

        return jsonify({'results': results})

    except Exception as e:
        # Return error in JSON if something goes wrong
        return jsonify({'error': str(e)}), 400

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
