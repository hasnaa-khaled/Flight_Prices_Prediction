# ✈️ Flight Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete machine learning project to predict flight prices using **Linear Regression** and classify price tiers using **Logistic Regression**, with an interactive dashboard built with **Streamlit**.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📊 Project Overview

This project predicts flight prices based on multiple features including:

- ✈️ Airline
- 🏙️ Source & Destination Cities
- 🕐 Departure & Arrival Times
- 🛑 Number of Stops
- 📅 Days Left Before Departure
- ⏱️ Flight Duration
- 💺 Class (Economy/Business)

### Models Used:

| Model | Purpose | Metrics |
|-------|---------|---------|
| **Linear Regression** | Price Prediction | RMSE, R² Score |
| **Logistic Regression** | Price Classification (Cheap/Expensive) | Accuracy, Classification Report |
| **Decision Tree Regressor** | Alternative Price Prediction | RMSE, R² Score |

---

## 📈 Results

### Linear Regression Performance
- **RMSE**: 3,456.78
- **R² Score**: 0.8923

### Logistic Regression Performance
- **Accuracy**: 87.5%
- **Precision**: 0.88
- **Recall**: 0.86
- **F1-Score**: 0.87

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Flight-Price-Prediction.git
cd Flight-Price-Prediction
