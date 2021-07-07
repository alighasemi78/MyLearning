# Simple Linear Regression

## Intuition

### Formula

Simple regression is a simple linear functions as follows:

![simple regression formula](simpleReg.png)

The variables and coefficients are:
* y: Dependent Variable (DV). The variable we want to explain and predict.
* x1: Independent Variable (IV). The variable that has effect on the DV and changes it. It might not directly change it but there is some kind of association between the two.
* b1: Coefficient. Determines how a unit change in x1 effects a unit change in y.
* b0: Constant.

Let's look at an example:

![chart](chart-min.PNG)

We want to understand how salary depends on people's experience. The red dots are some observations that we have. Simple regression draws the black line that tries to fit the data in the best way. The Constant mean the point where the line crosses the y axes.

![chart2](chart2-min.PNG)

It means when experience is 0, salary is 30,000 dolors. b1 is the slope of the line. The steeper the line, greater b1 is.

![chart3](chart3-min.PNG)

### Best Fitting Line

![chart4](chart4-min.PNG)

The model line shows us where these red crosses should be according to the model. "y" is what the observation shows us and "y^" is what the model predicts where "y" should be. The green line between them is the distance of "y" and "y^". Simple linear regression draws a lot of these lines for all the observations and calculate the distance with ( y - y^ )^2^ and then calculate the sum of this formula for all of the observations and finally find the minimum of these sums for different model lines. This method is called **Ordinary Least Squares**.

## Practical

### Training the Simple Linear Regression model on the Training set

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X=X_train, y=y_train)
```

### Predicting the Test set results

```python
y_pred = regressor.predict(X=X_test)
```

### Visualising the Training set results

```python
plt.scatter(x=X_train, y=y_train, color="red")
plt.plot(X_train, regressor.predict(X=X_train), color="blue")
plt.title(label="Salary vs Experience (Training set)")
plt.xlabel(xlabel="Years of Experience")
plt.ylabel(ylabel="Salary")
plt.show()
```

![training vis](vis_train.png)

### Visualising the Test set results

```python
plt.scatter(x=X_test, y=y_test, color="red")
plt.plot(X_test, y_pred, color="blue")
plt.title(label="Salary vs Experience (Test set)")
plt.xlabel(xlabel="Years of Experience")
plt.ylabel(ylabel="Salary")
plt.show()
```

![training vis](vis_test.png)

The thing to note here is that the plot functions work the same with both X_train and regressor.predict(X_train). That is because the regressor was fitted on one dataset and has created a single unique equation.