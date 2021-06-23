import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
cancer=pd.read_csv('cancer_data.csv')
cancer.head(2)
sns.countplot(x=’diagnosis’,data=cancer)
print(‘total no. of patients = ’,len(cancer.index))
sns.countplot(x=’diagnosis’,hue=’ area_se’,data=cancer)
sns.countplot(x=’diagnosis’,hue=’ perimeter_se’,data=cancer)
sns.countplot(x=’diagnosis’,hue=’ concavity_mean’,data=cancer)
sns.countplot(x=’diagnosis’,hue=’ fractal_dimension_mean’,data=cancer)
sns.countplot(x=’diagnosis’,hue=’ radius_mean’,data=cancer)
cancer.head(2)
cancer.isnull.sum()
sns.heatmap(cancer.isnull())
X=cancer.drop([‘diagnosis’],axis=1)
Y=cancer[‘diagnosis’]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)
From sklearn.linear_model import LogisticRegression
Lmodl=LogisticRegression()
Lmodl.fit(X_train,y_train)
X_test	
Y_test
Y_pred=lmodl.predict(X_test)
Y_pred
Y_test
From sklearn.metrices.import accuracy_score
Acc=lmodl.score(X_test,y_test)
Print(‘The logistic regression model score =’,acc*100)