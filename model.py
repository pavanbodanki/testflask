import pandas as pd
import numpy as np
import joblib,requests,json
df=pd.read_csv('Iris.csv')
df.head()
df.describe()
df.info()
X=df.iloc[:,:5]
y=df.iloc[:,5]
print(X)
print(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
lr= LogisticRegression(random_state = 42, max_iter = 1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
df.head()
df.drop(['Id'],axis=1)
import seaborn as sns
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
sns.heatmap(df.corr(), annot = True)
X = df.drop(columns = 'Species')
y = df['Species']
y_pred
y_pred = le.inverse_transform(y_pred)
y_pred


joblib.dump(lr, open('model.pkl','wb'))
model =joblib.load(open('model.pkl','rb'))
