# Polynomial Regression

## Intuition

The polynomial regression formula is as follows:

![formula](formula.png)

Let's say we got a dataset looking like this:

![chart1](chart-min.PNG)

If we try to use simple linear regression here, it will not fit quite well. That is why we use a polynomial regression and in this case fits perfectly:

![chart2](chart2-min.PNG)

The curve comes from the powered variables.

### Why still linear?

When we talk about linear or not, we do not talk about the variables; we talk about the coefficients. As can be seen, the coefficients have a linear relation. The reason of the importance of the coefficients is that our final goal is to find the coefficient.

## Practical

### Training the Linear Regression model on the whole dataset

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X=X, y=y)
```

### Training the Polynomial Regression model on the whole dataset

```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X=X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X=X_poly, y=y)
```

### Visualising the Linear Regression results

```python
plt.scatter(x=X, y=y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title(label="Truth or Bluff (Linear Regression)")
plt.xlabel(xlabel="Position Level")
plt.ylabel(ylabel="Salary")
plt.show()
```

![linear-vis](linear-vis.png)

### Visualising the Polynomial Regression results

```python
plt.scatter(x=X, y=y, color="red")
plt.plot(X, lin_reg_2.predict(X_poly), color="blue")
plt.title(label="Truth or Bluff (Polynomial Regression)")
plt.xlabel(xlabel="Position Level")
plt.ylabel(ylabel="Salary")
plt.show()
```

![poly-vis](poly-vis.png)

