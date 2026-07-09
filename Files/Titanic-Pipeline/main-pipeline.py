import pandas as pd
import os
import pickle
from prefect import task, flow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

@task
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

@task
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    print(f"Rows with null values: {df.isnull().sum().sum()}")
    return df

@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df = pd.get_dummies(df, columns=['Embarked'], dtype=int)

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Alone'] = (df['Family_size'] == 1).astype(int)

    print(f"Features after engineering: {df.columns.tolist()}")
    return df

@task
def save_features(df: pd.DataFrame):
    os.makedirs("outputs/dataset", exist_ok=True)
    df.to_csv("outputs/dataset/final_features.csv", index=False)
    print("Features saved")

@task
def train_test_model(df: pd.DataFrame):
    forest = RandomForestClassifier(random_state=42)

    x = df.drop(columns=['Survived'])
    y = df['Survived']

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    forest.fit(xtrain, ytrain)

    ypred = forest.predict(xtest)
    acc = accuracy_score(ytest, ypred)

    print(f"Model trained with {acc} score")
    return forest, acc

@task
def save_model(model, acc):
    os.makedirs("model", exist_ok=True)

    with open("model/model_forest.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved with acc score {acc}")

@flow(name="Titanic Flow pipeline")
def titanic_flow(path: str):
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    save_features(df)
    model, accuracy = train_test_model(df)
    save_model(model, accuracy)

if __name__ == "__main__":
    titanic_flow("D:/PROGRAMMING/AI/Datasets/Titanic.csv")