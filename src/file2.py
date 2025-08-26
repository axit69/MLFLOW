import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='axit69', repo_name='MLFLOW', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/axit69/MLFLOW.mlflow')


wine = load_wine()

X= wine.data
y= wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

max_depth = 10
n_estimator = 8

# mention your experiment below by default it will be running on default experiment
# if experiemnt does not exists then it will create a new experiemnt

mlflow.set_experiment('Mlops-Project-1')

# other than that if we want to set a experiemnt then we can also do it with the help of experiment id as well 
with mlflow.start_run():
    
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
 
    accuracy = accuracy_score(y_test, y_pred)

    # to log or to capture the parameters in mlflow

    

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimator)

    # creating a confusion matrix plot

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=[6,6])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig('Confusion_matrix.png')

    # log artifacts using mlflow

    mlflow.log_artifact('Confusion_matrix.png')
    mlflow.log_artifact(__file__)

    
    # log the model
    mlflow.sklearn.log_model(sk_model=rf, artifact_path= 'Random-Forest-Model')

    # Tags
    mlflow.set_tags({"Author": 'Axit', "Project": 'Wine classification'})
    

    print(accuracy)
