import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# Add current dir to path to import model architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_architecture import extract_image_features

def run_inference(image_path, methane_ppm_left=15.0, methane_ppm_right=16.0, temp1=29.5, temp2=31.5):
    """
    Performs single-image live inference purely from visual and sensor modalities.
    """
    print(f"[*] Processing single IR image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Extract dynamic 128-d image feature vector
    img_vector = extract_image_features(image_path)

    # 2. Prepare features (128 img + 2 meta temps + 5 gas = 135 total features)
    meta_features = np.array([float(temp1), float(temp2)])
    gas_features = np.array([
        float(methane_ppm_left),
        float(methane_ppm_right),
        0.01, # ppm_left_std placeholder
        0.02, # ppm_right_std placeholder
        float(methane_ppm_left - methane_ppm_right) # ppm_delta_mean
    ])

    combined_features = np.concatenate([img_vector, meta_features, gas_features]).reshape(1, -1)

    # 3. Load model bundle
    bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.pkl")
    if not os.path.exists(bundle_path):
        print("[!] Model bundle not found. Please run train_predictor.py first.")
        return None

    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    # 4. Transform and Predict
    scaled_input = bundle['scaler_X'].transform(combined_features)
    prediction = bundle['model'].predict(scaled_input, verbose=0)[0]

    # Map raw model output thresholds
    s50, s75, s90, s95 = prediction

    # 5. Compute dynamic freshness score based strictly on predicted remaining days to 95% terminal decay (0 to 6 days scale)
    s95_clamped = max(0.0, min(float(s95), 6.0))
    freshness_score = float((s95_clamped / 6.0) * 100.0)

    print(f"[*] Results: Freshness {freshness_score:.1f}%, 95% Spoilage remaining: {s95_clamped:.2f} days")
    return {
        'freshness_score': freshness_score,
        'spoil_50_days': max(0.0, float(s50)),
        'spoil_75_days': max(0.0, float(s75)),
        'spoil_90_days': max(0.0, float(s90)),
        'spoil_95_days': s95_clamped
    }
