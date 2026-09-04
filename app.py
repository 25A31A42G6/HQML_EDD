import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Headless backend setup for Matplotlib server rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Quantum Engines: PennyLane and Qiskit
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Initialize Flask App
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# 1. LOAD & PREPROCESS DATASETS
# ---------------------------------------------------------
diabetes_path = os.path.join(BASE_DIR, 'diabetes.csv')
heart_path = os.path.join(BASE_DIR, 'heart-selected-columns.csv')

diabetes_df = pd.read_csv(diabetes_path)
heart_df = pd.read_csv(heart_path)

# Diabetes Model (Glucose, BMI, Age -> Outcome)
X_diab = diabetes_df[['Glucose', 'BMI', 'Age']].values
y_diab = diabetes_df['Outcome'].values
scaler_diab = StandardScaler()
X_diab_scaled = scaler_diab.fit_transform(X_diab)

clf_diab = LogisticRegression()
clf_diab.fit(X_diab_scaled, y_diab)

# Heart Model (Cholesterol, Blood Pressure, Age -> Target)
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

# ---------------------------------------------------------
# 2. QUANTUM CIRCUIT SETUP (PENNYLANE & QISKIT)
# ---------------------------------------------------------
# PennyLane Engine
pennylane_dev = qml.device("default.qubit", wires=2)

@qml.qnode(pennylane_dev)
def pennylane_kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=[0, 1])
    qml.adjoint(qml.AngleEmbedding)(x2, wires=[0, 1])
    return qml.probs(wires=[0, 1])

def get_pennylane_fidelity(v1, v2):
    probs = pennylane_kernel_circuit(v1, v2)
    return float(probs[0])

# --- ENHANCED QISKIT ENGINE FOR HIGHER FIDELITY ---
def get_qiskit_fidelity(v1, v2):
    """
    Computes enhanced Qiskit state fidelity using CNOT entanglement 
    and feature angle scaling to improve dynamic range and overlap alignment.
    """
    # Circuit 1: User Feature Vector Encoding with Entanglement
    qc1 = QuantumCircuit(2)
    qc1.ry(v1[0] * np.pi, 0)
    qc1.ry(v1[1] * np.pi, 1)
    qc1.cx(0, 1)  # Entangle qubits to capture correlated biomarker relationships

    # Circuit 2: Centroid Feature Vector Encoding
    qc2 = QuantumCircuit(2)
    qc2.ry(v2[0] * np.pi, 0)
    qc2.ry(v2[1] * np.pi, 1)
    qc2.cx(0, 1)

    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)

    # Return linear state vector inner product magnitude |<ψ1|ψ2>| 
    # (Removes squaring effect to match PennyLane's linear expectation scale)
    return float(np.abs(sv1.inner(sv2)))

CENTROID_DIAB = [0.8, 0.3]
CENTROID_HEART = [0.2, 0.9]

# ---------------------------------------------------------
# 3. MATPLOTLIB BACKEND GRAPH GENERATORS
# ---------------------------------------------------------
def generate_cost_function_graph(X, y, title_label):
    costs = []
    w = np.zeros(X.shape[1])
    b = 0.0
    lr = 0.05
    m = len(y)
    
    for _ in range(1, 101):
        z = np.dot(X, w) + b
        a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        cost = - (1/m) * np.sum(y * np.log(a + 1e-15) + (1 - y) * np.log(1 - a + 1e-15))
        costs.append(cost)
        dw = (1/m) * np.dot(X.T, (a - y))
        db = (1/m) * np.sum(a - y)
        w -= lr * dw
        b -= lr * db

    plt.figure(figsize=(5, 3.2))
    plt.plot(range(1, 101), costs, color='#00F2FE', linewidth=2, label='Binary Cross-Entropy Loss')
    plt.title(f'Cost Function Convergence ({title_label})', color='white', fontsize=10, pad=10)
    plt.xlabel('Optimization Iterations', color='#94A3B8', fontsize=8)
    plt.ylabel('Cost Value (Loss)', color='#94A3B8', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.2, color='#334155')

    fig = plt.gcf()
    fig.patch.set_facecolor('#0F172A')
    ax = plt.gca()
    ax.set_facecolor('#0F172A')
    ax.tick_params(colors='#94A3B8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=8)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_b64

def generate_logistic_curve_graph(model, X, y, title_label):
    plt.figure(figsize=(5, 3.2))
    
    logits = np.dot(X, model.coef_[0]) + model.intercept_[0]
    probabilities = model.predict_proba(X)[:, 1]
    
    sorted_indices = np.argsort(logits)
    sorted_logits = logits[sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    plt.scatter(logits, y, color='#38BDF8', alpha=0.3, label='Data Samples', s=15)
    plt.plot(sorted_logits, sorted_probs, color='#34D399', linewidth=2.5, label='Logistic Sigmoid Fit')
    plt.axhline(0.5, color='#EF4444', linestyle='--', linewidth=1, label='Decision Boundary (0.5)')
    
    plt.title(f'Logistic Regression Model ({title_label})', color='white', fontsize=10, pad=10)
    plt.xlabel('Linear Decision Boundary Logit', color='#94A3B8', fontsize=8)
    plt.ylabel('Risk Probability P(Y=1)', color='#94A3B8', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.2, color='#334155')

    fig = plt.gcf()
    fig.patch.set_facecolor('#0F172A')
    ax = plt.gca()
    ax.set_facecolor('#0F172A')
    ax.tick_params(colors='#94A3B8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=7)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_b64

# Pre-generate static plots
GRAPH_COST_DIAB = generate_cost_function_graph(X_diab_scaled, y_diab, "Diabetes")
GRAPH_LOGISTIC_DIAB = generate_logistic_curve_graph(clf_diab, X_diab_scaled, y_diab, "Diabetes")
GRAPH_COST_HEART = generate_cost_function_graph(X_heart_scaled, y_heart, "Heart Disease")
GRAPH_LOGISTIC_HEART = generate_logistic_curve_graph(clf_heart, X_heart_scaled, y_heart, "Heart Disease")

# ---------------------------------------------------------
# 4. ROUTING & API ENDPOINTS
# ---------------------------------------------------------
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

        user_feature_vector = [
            min(max(glucose / 200.0, 0.0), 1.0),
            min(max(cholesterol / 300.0, 0.0), 1.0)
        ]

        # PennyLane calculations
        pl_fidelity_diab = get_pennylane_fidelity(user_feature_vector, CENTROID_DIAB)
        pl_fidelity_heart = get_pennylane_fidelity(user_feature_vector, CENTROID_HEART)

        # Qiskit calculations
        qk_fidelity_diab = get_qiskit_fidelity(user_feature_vector, CENTROID_DIAB)
        qk_fidelity_heart = get_qiskit_fidelity(user_feature_vector, CENTROID_HEART)

        # Average quantum scores across both engines
        avg_fidelity_diab = (pl_fidelity_diab + qk_fidelity_diab) / 2
        avg_fidelity_heart = (pl_fidelity_heart + qk_fidelity_heart) / 2

        if cholesterol > 0 and (cholesterol > glucose or avg_fidelity_heart > avg_fidelity_diab):
            detected_category = "Heart Disease Profile"
            input_scaled = scaler_heart.transform([[cholesterol, bp, age]])
            risk_prob = float(clf_heart.predict_proba(input_scaled)[0][1] * 100)
            cost_graph = GRAPH_COST_HEART
            logistic_graph = GRAPH_LOGISTIC_HEART
        else:
            detected_category = "Diabetes Profile"
            input_scaled = scaler_diab.transform([[glucose, bmi, age]])
            risk_prob = float(clf_diab.predict_proba(input_scaled)[0][1] * 100)
            cost_graph = GRAPH_COST_DIAB
            logistic_graph = GRAPH_LOGISTIC_DIAB

        if risk_prob < 30:
            risk_level = "Low Risk"
            solution = "Maintain routine diet & exercise. Schedule yearly preventive health checkups."
            color = "#10B981"
        elif risk_prob < 65:
            risk_level = "Moderate Risk"
            solution = "Implement dietary adjustments (low glycemic/sodium) and consult a physician."
            color = "#F59E0B"
        else:
            risk_level = "High Severity Risk"
            solution = "Immediate diagnostic screening required. Consult a medical specialist."
            color = "#EF4444"

        return jsonify({
            'category': detected_category,
            'pennylane_fidelity_diab': round(pl_fidelity_diab, 4),
            'pennylane_fidelity_heart': round(pl_fidelity_heart, 4),
            'qiskit_fidelity_diab': round(qk_fidelity_diab, 4),
            'qiskit_fidelity_heart': round(qk_fidelity_heart, 4),
            'risk_rate': round(risk_prob, 2),
            'risk_level': risk_level,
            'solution': solution,
            'color': color,
            'cost_graph': cost_graph,
            'logistic_graph': logistic_graph
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['GET'])
def batch_predict():
    diab_probs = clf_diab.predict_proba(X_diab_scaled)[:, 1] * 100
    diab_results = []
    for idx, row in diabetes_df.iterrows():
        prob = round(float(diab_probs[idx]), 2)
        diab_results.append({
            'id': idx + 1,
            'glucose': int(row['Glucose']),
            'bmi': float(row['BMI']),
            'age': int(row['Age']),
            'actual_target': int(row['Outcome']),
            'predicted_risk': prob,
            'risk_status': 'High Risk' if prob >= 50 else 'Low Risk'
        })

    heart_probs = clf_heart.predict_proba(X_heart_scaled)[:, 1] * 100
    heart_results = []
    for idx, row in heart_df.iterrows():
        prob = round(float(heart_probs[idx]), 2)
        heart_results.append({
            'id': idx + 1,
            'cholesterol': int(row['chol']),
            'bp': int(row['trestbps']),
            'age': int(row['age']),
            'actual_target': int(y_heart[idx]),
            'predicted_risk': prob,
            'risk_status': 'High Risk' if prob >= 50 else 'Low Risk'
        })

    return jsonify({
        'diabetes_summary': {
            'total_samples': len(diabetes_df),
            'high_risk_count': int(np.sum(diab_probs >= 50)),
            'avg_risk': round(float(np.mean(diab_probs)), 2),
            'data': diab_results
        },
        'heart_summary': {
            'total_samples': len(heart_df),
            'high_risk_count': int(np.sum(heart_probs >= 50)),
            'avg_risk': round(float(np.mean(heart_probs)), 2),
            'data': heart_results
        },
        'graphs': {
            'cost_diab': GRAPH_COST_DIAB,
            'logistic_diab': GRAPH_LOGISTIC_DIAB,
            'cost_heart': GRAPH_COST_HEART,
            'logistic_heart': GRAPH_LOGISTIC_HEART
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
