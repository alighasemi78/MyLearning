# Support Vector Regression (SVR)

## Intuition

Imagine one dataset. In simple linear regression we want to find the line that best fits the dataset. In SVR we want to find a tube that best fits the dataset:

![chart](chart-min.PNG)

In SVR we do not care about the points that fall in the tube. That is the key behind SVR which it gives our dataset a margin of error. But we do care about the points that fall out of the tube. We calculate the distance of these points with the tube not the line. These points are called **Slack Variables**.
The slack variables are vectors and they are the ones that dictate the formation of the tube. That is the reason behind the name **Support Vectors**.

## Practical

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X=X)
y = sc_y.fit_transform(X=y)
```

### Training the SVR model on the whole dataset

```python
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X=X, y=y)
```

### Predicting a new result

```python
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
```

### Visualising the SVR results

```python
plt.scatter(x=sc_X.inverse_transform(X), y=sc_y.inverse_transform(y), color="red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color="blue")
plt.title(label="Truth or Bluff (Support Vector Regression)")
plt.xlabel(xlabel="Position Level")
plt.ylabel(ylabel="Salary")
plt.show()
```

![vis](vis.png)