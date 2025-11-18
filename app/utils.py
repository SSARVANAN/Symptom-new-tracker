import os
import joblib
import pandas as pd

# -------------------------------
# Path to the artifacts folder
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, 'artifacts', 'pipeline.joblib')


def load_pipeline():
    """
    Load the trained pipeline and label encoder from artifacts.
    Returns:
        pipeline: scikit-learn Pipeline object
        le: LabelEncoder object
    """
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")
    
    artifacts = joblib.load(PIPELINE_PATH)
    pipeline = artifacts['pipeline']
    le = artifacts['label_encoder']
    return pipeline, le


def prepare_input(payload, pipeline):
    """
    Convert JSON payload into DataFrame for prediction and fill missing symptoms.
    Args:
        payload: dict containing symptom values
        pipeline: trained scikit-learn pipeline
    Returns:
        df_input: pandas DataFrame ready for prediction
    """
    df_input = pd.DataFrame([payload])

    # Fill missing features with 0 (like in app.py)
    feature_cols = pipeline.named_steps['model'].feature_names_in_
    for col in feature_cols:
        if col not in df_input:
            df_input[col] = 0

    # Ensure correct column order
    df_input = df_input[feature_cols]

    return df_input


def make_prediction(df_input, pipeline, le):
    """
    Make predictions and return results with probabilities.
    Args:
        df_input: DataFrame prepared for prediction
        pipeline: trained scikit-learn pipeline
        le: LabelEncoder for decoding predictions
    Returns:
        List of dicts with disease and probability
    """
    preds = pipeline.predict(df_input)
    proba = pipeline.predict_proba(df_input)

    disease_preds = le.inverse_transform(preds)
    results = []
    for disease, prob in zip(disease_preds, proba):
        results.append({'disease': disease, 'probability': float(max(prob))})

    return results

