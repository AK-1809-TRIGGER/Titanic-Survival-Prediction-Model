# 🚢 Titanic Survival Prediction

> My first end-to-end Machine Learning project — from raw data exploration to a deployed Streamlit app, containerized with Docker.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://titanic-survival-prediction-model-2bjpwmpxsdntkeek9bjalj.streamlit.app/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Pull%20Image-2496ED?style=for-the-badge&logo=docker)](https://hub.docker.com/r/abhishek1255/titanic-survival)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?style=for-the-badge&logo=github)](https://github.com/AK-1809-TRIGGER/Titanic-Survival-Prediction-Model)
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/5819bad3-75ec-49f1-9470-4f2390c793c5" />

---

## 🎯 What It Does

Enter a passenger's details — class, age, gender, fare — and the model predicts whether they would have survived the Titanic disaster.

**Model Accuracy: 77%** using Logistic Regression with custom feature engineering.

🔗 **[Try the live app here](https://titanic-survival-prediction-model-2bjpwmpxsdntkeek9bjalj.streamlit.app/)**

---

## 🔍 Key Insights from EDA

| Finding | Survival Rate |
|---------|--------------|
| Female passengers | ~74% |
| Male passengers | ~19% |
| 1st class vs 3rd class | Significantly higher |
| Children (under 16) | Higher than average |

These patterns reflect the real "women and children first" lifeboat protocol — and they directly shaped my feature engineering decisions.

---

## ⚙️ Feature Engineering

Two custom features built from EDA insights:

- **`IsChild`** — `True` if passenger age < 16
- **`Priority`** — `True` if passenger was female OR a child (mirrors actual lifeboat boarding priority)

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Model | Logistic Regression + StandardScaler |
| Visualization | Seaborn, Matplotlib |
| Frontend | Streamlit |
| Serialization | Joblib |
| Containerization | Docker |

---

## 📁 Project Structure

```
Titanic-Survival-Prediction-Model/
├── train.csv               # Kaggle training dataset
├── titanic.py              # Trains model → saves model.pkl + scaler.pkl
├── titanic.ipynb           # EDA notebook
├── streamlit_app.py        # Streamlit UI (self-contained, no API needed)
├── model.pkl               # Saved trained model
├── scaler.pkl              # Saved StandardScaler
├── requirements.txt        # Python dependencies
└── Dockerfile              # For Docker deployment
```

---

## 🚀 Run with Docker

```bash
# Pull the pre-built image
docker pull abhishek1255/titanic-survival:latest

# Run the app
docker run -p 8501:8501 abhishek1255/titanic-survival:latest
```

Open `http://localhost:8501` in your browser.

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/AK-1809-TRIGGER/Titanic-Survival-Prediction-Model.git
cd Titanic-Survival-Prediction-Model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model
python titanic.py

# 4. Launch the app
streamlit run streamlit_app.py
```

---

## 🔮 What I'd Improve Next

- [ ] Try Random Forest or XGBoost and compare accuracy
- [ ] Add family size feature: `FamilySize = SibSp + Parch + 1`
- [ ] Log-transform Fare to handle extreme outliers
- [ ] Hyperparameter tuning with GridSearchCV

---

## 📊 Dataset

Kaggle Titanic Competition — [Download train.csv](https://www.kaggle.com/competitions/titanic/data)

---

## 👤 Author

**Abhishek** (AK-1809-TRIGGER)
Started with Data Analysis → now building ML projects.

[GitHub](https://github.com/AK-1809-TRIGGER) · [Docker Hub](https://hub.docker.com/r/abhishek1255)

---

*If this helped you, leave a ⭐ — it means a lot for a first ML project!*
