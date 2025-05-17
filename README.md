# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer.

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

8.Find the accuracy of the model.

9.End the Program.

## Program:

Program to implement the SVM For Spam Mail Detection..

Developed by: SACHIN S

RegisterNumber: 212224040283 

```python
import pandas as pd

data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![{D4C219A7-BA66-460A-A866-2AB44ED4EA63}](https://github.com/user-attachments/assets/3ca469ac-36c2-45c0-b7b8-5ee00173e3e5)

![{A06AEE12-823F-4B0B-82D0-758186BD2806}](https://github.com/user-attachments/assets/0e7545a2-04ab-4c07-9133-b6105c4b5ac3)

![{26E51070-3517-482E-A91B-DB6D2CBE67E8}](https://github.com/user-attachments/assets/03e3a7fc-e166-4e59-aaad-7da5e0085a64)

![{D894EBBF-2C5F-4011-A280-405912254C6D}](https://github.com/user-attachments/assets/50611e62-b0f2-4ee8-900a-d42d0f995286)

![{E6A289ED-6DB2-44E3-A3CD-EBCF6CA59B04}](https://github.com/user-attachments/assets/4a2a6e36-68bc-4c65-8339-a6a2a8c9de49)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
