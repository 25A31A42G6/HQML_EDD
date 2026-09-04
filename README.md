# Skopeo Tech — QML Disease Risk & Clinical Assessment Platform

A hybrid Quantum-Classical Machine Learning web engine designed for early-stage disease risk stratification. Built for the **Smart India Hackathon (Problem Statement: SIH26139)**, this application leverages **PennyLane Quantum State Fidelity Clustering** to categorize patient clinical profiles into health domains (Diabetes vs. Heart Disease) and evaluates probability risk rates using **Logistic Regression**.

---

## Key Features

* **Quantum Kernel Clustering (QML):** Uses a 2-qubit quantum state overlap circuit (`default.qubit` simulator via PennyLane) to map non-linear biomarker interactions into Hilbert space.
* **Dynamic Patient Routing:** Evaluates clinical parameters (Glucose, Cholesterol, BMI, Blood Pressure) to automatically assign profiles to the correct disease dataset.
* **Logistic Regression Risk Calculation:** Predicts risk percentages based on preprocessed healthcare datasets (`diabetes.csv` and `heart-selected-columns.csv`).
* **Interactive Diagnostic Dashboard:** Styled with Tailwind CSS and powered by Chart.js for real-time risk gauge rendering and clinical solution recommendation routing.

---

## Project Structure

```text
HQML/
├── app.py                         # Flask backend & PennyLane QML engine
├── diabetes.csv                   # Diabetes clinical dataset
├── heart-selected-columns.csv     # Cardiovascular clinical dataset
├── README.md                      # Documentation
└── templates/
    └── index.html                 # Tailwind CSS & Chart.js frontend dashboard
