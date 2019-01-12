## Linear Regression in Tensorflow
Linear Regression and logistic regression are the simplest form of Machine Learning algorithms for 
predication & classification. Regression is a good place to start if you like to understand the 
intuition behind Neural Networks. Regression can be thought of as modeling a single neuron.


Linear regression identifies a linear relationship between two variable set(s). Given two set of 
variables X :{x1, x2,....xn} & Y: {y1, y2....yn} a linear relationship between the two can be 
defined in terms of slope 'm' of the line that passes thru all the points {(x1,y1), (x2,y2),...(xn, yn)} 
in the X, Y plane.

 ![alt text](https://www.kullabs.com/img/note_images/eA7pSlMa8FIftHQe.jpg)

Once the slope m and the intercept c is identified, the task of predicting yi for any given x is simple. 
You only need to substitute xi in the equation and get a corresponding yi. The values of m and c define
a model for the given data set.

In real world scenarios the line will not pass thru all the points, so the optimal line is the one 
that minimizes the mean square error of the perpendicular distance of each point from the line.

 ![alt text](https://i.stack.imgur.com/cj8j6.png)

The mean square error is computed by the formula.

![alt text](https://i.stack.imgur.com/19Cmk.gif)
When X is two-dimensional data, y is predicted on a plane given by the equation y = m1 * x1 + m2 * x2 + c. 
This can be extended to more dimensions as y = MX + c where M, X  are vectors. Below is a Tensor flow 
implementation of Linear regression that just uses random generated points (x, y) X & Y to plot a line. 
The data is used to train a model that approximates the slope m & bias c.

In [code](Linear.py) m (slope) is denoted by W (for weights) and c is denoted by b (for bias). This implementation tries to fit a polynomial, finding the parameted using Gradient Descent

### MultiVariate Regression

Now this can be extended to multivariate regressions. Also included is a [simple 2 variable multivariate regression.](multivariate.py)
