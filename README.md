# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Algorithm
1. Initialize weights randomly. 
2. Compute predicted values. 
3. Compute gradient of loss function.
4. Update weights using gradient descent

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ashwath.M
RegisterNumber:  212223230023
*/


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2024-08-31 202338](https://github.com/user-attachments/assets/7d3becec-899b-4456-968d-4516103f155e)
![Screenshot 2024-08-31 202445](https://github.com/user-attachments/assets/dad71e31-8107-4687-b9c6-410a32a0ea0a)
![Screenshot 2024-08-31 202510](https://github.com/user-attachments/assets/559e9c86-0575-4359-b32d-8c7cecd52787)
![Screenshot 2024-08-31 202518](https://github.com/user-attachments/assets/057acaf3-3717-4ce0-80cc-9cf4b87b2b64)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
