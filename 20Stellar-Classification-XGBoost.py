import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("20-digitalskysurvey.csv")
columns_to_drop=["objid","specobjid","rerun","field","camcol","run"]
## gerekli olmayan sütunların yok edilmesi
df.drop(columns_to_drop,axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["class"]=le.fit_transform(df["class"])

## datayı anlama
#sns.pairplot(df,hue="class")

fig,axes = plt.subplots(nrows=1,ncols=3)
ax=sns.histplot(df[df["class"]==2].redshift,ax=axes[0])
ax=sns.histplot(df[df["class"]==0].redshift,ax=axes[1])
ax=sns.histplot(df[df["class"]==1].redshift,ax=axes[2])
#plt.show()
# burdan redshifti seçtik 

from sklearn.preprocessing import StandardScaler
X=df.drop("class",axis=1)
y=df["class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=15)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=100)
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
from sklearn.model_selection import GridSearchCV
params = {
    "learning_rate" : [0.01,0.1],
    "n_estimators":[100,200,300,500],
    "max_depth":[5,8,12,20,30],
    "colsample_bytree":[0.3,0.4,0.5,0.8,1]
}
grid_search=GridSearchCV(estimator=XGBClassifier(),param_grid=params,cv=5,n_jobs=-1,verbose=1)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
y_pred_hyper=grid_search.predict(X_test)
print(classification_report(y_pred_hyper,y_test))
print(confusion_matrix(y_pred_hyper,y_test))

