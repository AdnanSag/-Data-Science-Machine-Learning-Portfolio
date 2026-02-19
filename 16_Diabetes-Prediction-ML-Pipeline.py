import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

##diabetes dataset

df=pd.read_csv("datasets/16-diabetes.csv")
columns_to_check = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
X=df.drop("Outcome",axis=1)
y=df['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)

medians={}
for col in columns_to_check:
    median_value = X_train [ X_train[col] !=0][col].median()
    medians[col]=median_value
    X_train[col]=X_train[col].replace(0,median_value)
for col in columns_to_check:
    X_test[col]=X_test[col].replace(0,medians[col])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# modeller
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
def calculate_model_metrics(true,predicted):
    confusioMatrix = confusion_matrix(true,predicted)
    accuracyScore = accuracy_score(true,predicted)
    classificationReport=classification_report(true,predicted)
    return accuracyScore,confusioMatrix,classificationReport

models = {
    "rbf" :SVC(),
    "naive_bayes":GaussianNB(),
    "decision_tree":DecisionTreeClassifier(),
    "knn":KNeighborsClassifier(),
    "random_forest" :RandomForestClassifier()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_accuracy, model_train_confusion, model_train_report=calculate_model_metrics(y_train,y_train_pred) #modelin train accuracysi
    model_test_accuracy, model_test_confusion, model_test_report=calculate_model_metrics(y_test,y_test_pred) # modelin test accuracysi

    print( list(models.values())[i])
    print("Evaulation for Training Set")
    print("Accuracy :",model_train_accuracy)
    print("Confussion :",model_train_confusion)
    print("Report :",model_train_report)

    print("-----------------------------")
 
    print("Accuracy :",model_test_accuracy)
    print("Confussion :",model_test_confusion)
    print("Report :",model_test_report)

    print("-----------------------------")


#hyperparameter tunning

rf_params = {
    "n_estimators": [100, 200, 500],     
    "max_depth": [None, 5, 10, 20],     
    "min_samples_split": [2, 5, 10],      
    "min_samples_leaf": [1, 2, 4],       
    "max_features": ["sqrt", "log2"]
}

# 2. SVC (Support Vector Classifier) Parametreleri
svc_params = {
    "C": [0.1, 1, 10, 100],                
    "gamma": [1, 0.1, 0.01, 0.001, "scale"],
    "kernel": ["rbf", "linear", "sigmoid"]  
}

# 3. KNN (K-Nearest Neighbors) Parametreleri
knn_params = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],   
    "weights": ["uniform", "distance"],     
    "metric": ["euclidean", "manhattan"]    
}

from sklearn.model_selection import RandomizedSearchCV
randomcv_models = [
    ("KNN",KNeighborsClassifier(),knn_params),
    ("RF",RandomForestClassifier(),rf_params),
    ("SVC",SVC(),svc_params)
    ]

for name,model,params in randomcv_models:
    randomcv= RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=100,cv=3,n_jobs=-1)
    randomcv.fit(X_train,y_train)
    print("best params for :", name ,randomcv.best_params_)
"""
best params for : KNN {'weights': 'distance', 'n_neighbors': 9, 'metric': 'euclidean'}
best params for : RF {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
best params for : SVC {'kernel': 'rbf', 'gamma': 0.001, 'C': 100}
"""


models = {
    "rbf" :SVC(kernel="rbf",gamma=0.001,C=100),
    "knn":KNeighborsClassifier(n_neighbors=9,metric="euclidean",weights="distance"),
    "random_forest" :RandomForestClassifier(n_estimators=500,min_samples_split=5,min_samples_leaf=1,max_features="sqrt",max_depth=None)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_accuracy, model_train_confusion, model_train_report=calculate_model_metrics(y_train,y_train_pred) #modelin train accuracysi
    model_test_accuracy, model_test_confusion, model_test_report=calculate_model_metrics(y_test,y_test_pred) # modelin test accuracysi

    print( list(models.values())[i])
    print("Evaulation for Training Set")
    print("Accuracy :",model_train_accuracy)
    print("Confussion :",model_train_confusion)
    print("Report :",model_train_report)

    print("-----------------------------")
 
    print("Accuracy :",model_test_accuracy)
    print("Confussion :",model_test_confusion)
    print("Report :",model_test_report)

    print("-----------------------------")
