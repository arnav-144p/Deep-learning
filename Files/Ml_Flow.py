import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x,y = load_wine(return_X_y=True)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    n_estimator = 100
    max_depth = 5

    mlflow.log_param("N-estimators", n_estimator)
    mlflow.log_param("Max-Depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)

    acc = accuracy_score(ytest, ypred)

    mlflow.log_metric("Accuracy", acc)
    mlflow.sklearn.log_model(model,name="RF-Model")

    print(f"Run complete with accuracy {acc}")