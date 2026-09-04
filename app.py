import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pennylane as qml

# Initialize Flask App with CORS support
app = Flask(__name__)
CORS(app)

# ==========================================
# 1. LOAD & PREPROCESS DATASETS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_path = os.path.join(BASE_DIR, 'diabetes.csv')
heart_path = os.path.join(BASE_DIR, 'heart-selected-columns.csv')

# Load CSV Files
diabetes_df = pd.read_csv(diabetes_path)
heart_df = pd.read_csv(heart_path)

# Train Diabetes Model (Glucose, BMI, Age -> Outcome)
X_diab = diabetes_df[['Glucose', 'BMI', 'Age']].values
y_diab = diabetes_df['Outcome'].values
scaler_diab = StandardScaler()
X_diab_scaled = scaler_diab.fit_transform(X_diab)

clf_diab = LogisticRegression()
clf_diab.fit(X_diab_scaled, y_diab)

# Train Heart Disease Model (Cholesterol, Blood Pressure, Age -> Risk)
X_heart = heart_df[['chol', 'trestbps', 'age']].values
if 'target' in heart_df.columns:
    y_heart = heart_df['target'].values
elif 'oldpeak' in heart_df.columns:
    y_heart = (heart_df['oldpeak'] > 1.0).astype(int).values
else:
    y_heart = np.zeros(len(heart_df))

scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

clf_heart = LogisticRegression()
clf_heart.fit(X_heart_scaled, y_heart)

# ==========================================
# 2. QUANTUM MACHINE LEARNING (QML) CLUSTERING
# ==========================================
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    """2-Qubit Quantum State Overlap (Fidelity) Circuit"""
    qml.AngleEmbedding(x1, wires=[0, 1])
    qml.adjoint(qml.AngleEmbedding)(x2, wires=[0, 1])
    return qml.probs(wires=[0, 1])

def get_quantum_fidelity(v1, v2):
    """Calculates state fidelity return scalar float"""
    probs = quantum_kernel_circuit(v1, v2)
    return float(probs[0])  # Overlap probability |00>

# Quantum Cluster Centroids in Hilbert Feature Space
CENTROID_DIAB = [0.8, 0.3]   # Metabolic / High Glucose Cluster Target
CENTROID_HEART = [0.2, 0.9]  # Cardiovascular / High Cholesterol Cluster Target

# ==========================================
# 3. ROUTING AND PREDICTION ENDPOINTS
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        glucose = float(data.get('glucose', 0))
        cholesterol = float(data.get('cholesterol', 0))
        age = float(data.get('age', 45))
        bmi = float(data.get('bmi', 25))
        bp = float(data.get('bp', 120))

        # 1. Map input vector into normalized Quantum State bounds
        user_feature_vector = [
            min(max(glucose / 200.0, 0.0), 1.0),
            min(max(cholesterol / 300.0, 0.0), 1.0)
        ]

        # 2. Execute Quantum Clustering via State Overlap (Fidelity)
        fidelity_diab = get_quantum_fidelity(user_feature_vector, CENTROID_DIAB)
        fidelity_heart = get_quantum_fidelity(user_feature_vector, CENTROID_HEART)

        # 3. Dynamic Model Routing based on QML Fidelity & Biomarker Priorities
        if cholesterol > 0 and (cholesterol > glucose or fidelity_heart > fidelity_diab):
            detected_category = "Heart Disease Profile"
            input_scaled = scaler_heart.transform([[cholesterol, bp, age]])
            risk_prob = float(clf_heart.predict_proba(input_scaled)[0][1] * 100)
        else:
            detected_category = "Diabetes Profile"
            input_scaled = scaler_diab.transform([[glucose, bmi, age]])
            risk_prob = float(clf_diab.predict_proba(input_scaled)[0][1] * 100)

        # 4. Generate Clinical Risk Solutions
        if risk_prob < 30:
            risk_level = "Low Risk"
            solution = "Maintain a healthy diet and routine exercise. Schedule annual checkups."
            color = "#10B981"  # Emerald Green
        elif risk_prob < 65:
            risk_level = "Moderate Risk"
            solution = "Adopt lifestyle interventions (low glycemic/sodium diet) and consult a physician."
            color = "#F59E0B"  # Amber/Yellow
        else:
            risk_level = "High Severity Risk"
            solution = "Immediate medical assessment required. Initiate detailed clinical diagnostic screening."
            color = "#EF4444"  # Red

        return jsonify({
            'category': detected_category,
            'quantum_fidelity_diab': round(fidelity_diab, 4),
            'quantum_fidelity_heart': round(fidelity_heart, 4),
            'risk_rate': round(risk_prob, 2),
            'risk_level': risk_level,
            'solution': solution,
            'color': color
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Running on port 5001 to avoid macOS AirPlay port conflicts
    app.run(debug=True, port=5001)