# Random Forest Regression

## Intuition

Random forset is a version of **Ensemble Learning**. Ensemble learning is when you take multiple algorithms or take one algorithm multiple times, and create something more powerful than the original.

Random forest steps:

1. Pick at random K data points from the training set.
2. Build the decision tree associated to these K data points.
3. Choose the number of trees you want to build and repeat steps 1 and 2.
4. For a new data point, make each one of your trees predict the value of y for the data point in question, and assign the new data point the average across all of the predicted y values.

## Practical