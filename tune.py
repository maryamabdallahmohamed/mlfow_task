import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import dagshub
dagshub.init(repo_owner='maryamabdallahmohamed', repo_name='mlfow_task', mlflow=True)
df = pd.read_csv('data/train_data.csv')

# Split the data into features and target
X = df.drop('alert', axis=1)
y = df['alert']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
n_estimators = [50, 100]
max_depth = [None, 10, 20]

with mlflow.start_run(run_name="RandomForest Hyperparameter Tuning"):
    for n in n_estimators:
        for depth in max_depth:
            with mlflow.start_run(nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
            
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", accuracy)
                
                print(f"n_estimators: {n}, max_depth: {depth}, accuracy: {accuracy:.4f}")