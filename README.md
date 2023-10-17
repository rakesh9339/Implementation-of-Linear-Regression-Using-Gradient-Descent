# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Steps involved

1.Data Preparation:
The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data:
Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model:
The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model:
Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model:
Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model:
If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data:
Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results:
Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAKESH JS
RegisterNumber:  212222230115
```
```python

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### 1.Placement Data
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/906881a8-378a-496f-87fa-e04a6b68b48c)
### 2.Salary Data
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/ac65ed31-9e75-4718-a5b0-f76b2bd4f4e6)
### 3. Checking the null function()
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/80254414-d19a-40d5-bcde-a5b2d2fdc320)
### 4.Data Duplicate
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/1b7713de-f9f6-4a4f-9d2a-1d3afd05bd75)
### 5.Print Data
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/e672b342-e3a3-4964-a1d9-f66d73e09f0d)
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/927a0e92-056d-4252-a9e6-1973a03922e6)
### 6.Data Status
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/6b5f9885-327f-49b8-805a-a9f2a20b99c0)
### 7.y_prediction array
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/981ddbed-7686-4f43-a896-09cb8725d9ad)
### 8.Accuracy value
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/d1978977-ab9a-4f59-a2fc-1944fdf34a8e)
### 9.Confusion matrix
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/2eb60e93-5b60-44bd-9dc3-8d7afc0ef62d)
### 10.Classification Report
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/6320a38a-dda5-4d3a-94b9-2d9dba8426a5)
### 11.Prediction of LR
![image](https://github.com/aldrinlijo04/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118544279/057a45e1-08e3-4de9-af44-9322010e588a)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
