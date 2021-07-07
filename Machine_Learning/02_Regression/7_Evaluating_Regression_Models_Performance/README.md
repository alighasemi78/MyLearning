# Evaluating Regression Models Performance

## R^2^

Remember when we talked about simple linear regression, we introduced a way to find the best fitting line. We had to find the distance of the real point from the line and square it and then sum all of these squared values and find the minimum. This value (SUM(y~i~ - y~i~\^)^2^) is called **Sum of Squared Residuals (SS~res~)**. Now think of the average line which goes horizontally through the middle of the points. This value ((y~i~ - y~avg~)^2^) is called the **Total Sum of Squares (SS~tot~)**. With these values, we can introduce **R^2^**:

![r2 formula](r2%20formula.png)

R^2^ show us how good our line is compared to the average line. The closer is is to 1 the better.

## Adjusted R^2^

Think of adding a new variable to our linear regression. So, it is a multiple linear regression now. There is a problem. Whenever we add a new variable, the R^2^ always gets greater. This means that our model is getting better and better. However, this is not right. We want to know how to improve our model. This is where **Adjusted R^2^** comes into place:

![adjusted r2 formula](adj%20r2%20formula.png)

Here p is the number of regressors (variables) and n is the sample size. As can be seen when we add a new variable, R^2^ increases so (1 - R^2^) decreases. Also the fraction increases. Moreover, if the effect of the variable addition is good for our model the Adjusted R^2^ will increase.