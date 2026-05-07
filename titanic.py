import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_and_save(csv_path='train.csv'):
    train = pd.read_csv(csv_path)

    # Drop unused columns
    train = train.drop(columns=['Ticket', 'Parch', 'PassengerId', 'Cabin', 'Name'])

    
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Fare'] = train['Fare'].fillna(train['Fare'].median())
    train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

    # Feature engineering
    train['IsChild'] = train['Age'] < 16
    #that new column help me to check is actually child(male,female)/female 
    train['Priority'] = ((train['Sex'] == 'female') | (train['IsChild']))
    #train['Priority'] = ~((train['Sex'] == 'male') & (train['Age'] > 18))
    #That Function 

    # Encode
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    X = train[['Pclass', 'Sex', 'Age', 'Fare', 'IsChild', 'Priority']]
    y = train['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #Use for Normalization to rescales features to a common range
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Model Accuracy: {acc:.2%}")

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved: model.pkl, scaler.pkl")

if __name__ == '__main__':
    train_and_save()