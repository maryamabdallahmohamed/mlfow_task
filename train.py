import mlflow
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import dagshub
dagshub.init(repo_owner='maryamabdallahmohamed', repo_name='mlfow_task', mlflow=True)
train=pd.read_csv('data/train_data.csv')

mlflow.set_experiment("Earthquake Alert Classification")


X_train, X_test, y_train, y_test = train_test_split(
    train.drop('alert', axis=1),
    train['alert'],
    test_size=0.2,
    random_state=42
)


model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    activation='relu',
    batch_size=32,
    random_state=42
)

with mlflow.start_run(run_name="MLPClassifier"):
    mlflow.set_tag("model_type", "Neural Network")
    mlflow.log_params({
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "batch_size": 32,
        "max_iter": 500,
        "random_state": 42
    })
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "nn_model.joblib")
    mlflow.log_artifact("nn_model.joblib") 