import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/15-gym_crowdedness.csv")
df["date"] = pd.to_datetime(df['date'],utc=True)
df["year"]= df["date"].dt.year
df.drop("date",axis=1,inplace=True)
df.drop("timestamp",axis=1,inplace=True)

#dependent independent features
X=df.drop("number_people",axis=1)
y=df['number_people']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

# modeller
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def calculate_model_metrics(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true,predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square=r2_score(true,predicted)
    return mae,rmse,r2_square

models = {
    "Linear Regression" : LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2=calculate_model_metrics(y_train,y_train_pred) #modelin train accuracysi
    model_test_mae, model_test_rmse, model_test_r2=calculate_model_metrics(y_test,y_test_pred) # modelin test accuracysi

    print( list(models.values())[i])
    print("Evaulation for Training Set")
    print("RMSE :",model_train_rmse)
    print("Mean Absolute Error :",model_train_mae)
    print("R2 Score :",model_train_r2)

    print("-----------------------------")
 
    print("RMSE :",model_test_rmse)
    print("Mean Absolute Error :",model_test_mae)
    print("R2 Score :",model_test_r2)
 
    print("-----------------------------")


#hyperparameter tunning

knn_params = {"n_neighbors": [2,3,10,20,40,50]}
rf_params = {
    "n_estimators":[100,200,500,1000],
    "max_depth" : [5,8,10,15,None],
    "max_features" : ["sqrt","log2",5,7,10],
    "min_samples_split":[2,8,12,20]
}

from sklearn.model_selection import RandomizedSearchCV
randomcv_models = [
    ("KNN",KNeighborsRegressor(),knn_params),
    ("RF",RandomForestRegressor(),rf_params)
    ]

for name,model,params in randomcv_models:
    randomcv= RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=100,cv=3,n_jobs=-1)
    randomcv.fit(X_train,y_train)
    print("best params for :", name ,randomcv.best_params_)

"""


"""

models = {
    "K-Neighbors Regressor" : KNeighborsRegressor(n_neighbors=2),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=500,
                                                     min_samples_split=2,
                                                     max_depth=None,
                                                     max_features=7)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2=calculate_model_metrics(y_train,y_train_pred) #modelin train accuracysi
    model_test_mae, model_test_rmse, model_test_r2=calculate_model_metrics(y_test,y_test_pred) # modelin test accuracysi

    print( list(models.values())[i])
    print("Evaulation for Training Set")
    print("RMSE :",model_train_rmse)
    print("Mean Absolute Error :",model_train_mae)
    print("R2 Score :",model_train_r2)

    print("-----------------------------")
 
    print("RMSE :",model_test_rmse)
    print("Mean Absolute Error :",model_test_mae)
    print("R2 Score :",model_test_r2)
 
    print("-----------------------------")
