# Decision Tree Regression

## Intuition

Imagine a dataset like below:

![chart](chart-min.png)

What happens when you run the decision tree algorithm is that the dataset will split up in segments. For instance, the dataset above will spilt up to these segments:

![chart2](chart2-min.png)

The algorithm continues to split until it knows that by splitting, the amount of information added to its knowledge is less than the minimum.
The reason it is called a decision tree is that when we do the splits, when a new point is added then we have to check some questions to find the predicted value.

![chart3](chart3-min.png)

How to predict the new value? When we know in which segment the new point will fall, then the value of this point is the average of the values of that segment.

## Practical

### Training the Decision Tree Regression model on the whole dataset

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
```

### Visualising the Decision Tree Regression results (higher resolution)

```python
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
```

![vis](vis.png)