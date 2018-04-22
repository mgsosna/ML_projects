# [IN PROGRESS]
# Multivariate linear regression via gradient descent

## 1. Abstract
This repository contains a function, `gd_lm`, that performs a linear regression via gradient descent on any-dimensional data. The arguments for `gd_lm` are as follows:
* `X`: input data
* `y`: output data
* `alpha`: the learning rate
* `n_iter`: the number of iterations to take for gradient descent
* `figure`: for 1- and 2-dimensional `X` data, should a figure be plotted?
* `stop_thresh`: when MSE improvement falls below this value, gradient descent stops
* `n_runs`: number of times to run gradient descent (each with different starting conditions)
* `full`: should the full MSE and coefficient trajectories be saved?

This repository also includes the following helper functions:
* `gen_data`: generate any-dimensional random data with a positive relationship (whose strength can be adjusted)
* `analytical_reg`: calculate the analytical solution for N-dimensional data, when N > 1 (similar to R's `lm` function, but easier for iterating in a parameter scan)
* `gen_preds`: generate model predictions on any-dimensional data, given a set of regression coefficients

[grad_desc_lm.R](grad_desc_lm.R) and [grad_desc_lm.py](grad_desc_lm.py) include all functions, and [grad_desc_demo.R](grad_desc_demo.R) includes code for visualizations.

## 2. Background
### 2.1 Regression
<img align="right" src="https://i.imgur.com/1ltmiKM.png"> How can we quantify the relationship between two variables? It's intuitive to understand that, say, the more I study, the better I do on the exam. You can take fifty students, track how much they study and what their exam score was, then plot it and get something like the plot on the right. We can clearly see that there is a positive trend: the more I study, the better I do. 

But how can we quantify this relationship? **_How much better_** should I expect to do on my exam **_for every additional hour_** I study? One way to answer this question is with [linear regression](https://en.wikipedia.org/wiki/Linear_regression). A regression is a model that takes in some continuous input (e.g. number of hours studied) and spits out a continuous output (e.g. exam score). So for 1.25 hours of studying I should get a 59%; for 3.87 hours of studying I should get an 81%, etc. A linear regression is just a line that you draw though the data. Here's the mathematical form:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}=\beta_0&space;&plus;&space;\beta_1x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}=\beta_0&space;&plus;&space;\beta_1x" title="\hat{y}=\beta_0 + \beta_1x" /></a>

It reads as "the model's estimate of `y` equals the intercept + the slope * `x`." The intercept is the expected exam score for someone who didn't study at all, and the slope is the change in exam score for each additional hour of studying. The intercept and the slope are called **coefficients.**

### 2.2 Mean squared error
How can we tell if our regression is a good fit for the data? We can draw lots of lines through our data, and most of them won't describe the data well. Of the plots below, for example, the left and middle linear regressions clearly don't describe how "number of hours studied" and "exam score" relate to each other. 

![](https://i.imgur.com/8G5SCBQ.png)

We can quantify **how bad** the regression is through something called [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error). This is a measure of the average residual - the average distance between the model's prediction (the score it thinks the student got, given the amount they studied) and the actual output (the score the student actually got). We square the residuals so that it doesn't matter if the model predicted a value lower than versus greater than the actual value. The equation for mean squared error is below:

<a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2" title="MSE = \frac{1}{N} \sum_{i=1}^{N}(\hat{y_i}-y_i)^2" /></a>

### 2.3 Method 1: analytical solution
So we have a way to measure how bad our regression is, but that's still avoiding the point: how do we find the values for our coefficients? Given some data, where should we set the intercept and the slope? It turns out that we can find the optimal solution - or get very close - for the coefficients using matrix multiplication, as shown below:

<a href="https://www.codecogs.com/eqnedit.php?latex=(\mathbf{X'}\mathbf{X}){^{-1}}\mathbf{X}y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\mathbf{X'}\mathbf{X}){^{-1}}\mathbf{X}y" title="(\mathbf{X'}\mathbf{X}){^{-1}}\mathbf{X}y" /></a>

`X` is a matrix of input values. For our simple example of hours studied versus exam score, our matrix would only have one column: hours studied. (Each row would be a different student's number of hours studied.) But we could run a regression with many more variables, such as *hours since student last ate*, *hours of sleep last night*, etc. The data for each of these additional variables would get their own column.


Inspired by [Andrew Ng](http://www.andrewng.org/)'s machine learning Coursera course, I decided to write a function that performs linear regression via [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).




**To optimize the model intercept, iterate this equation:** <br><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)" title="\theta_0(t) = \theta_0(t-1) - \alpha \frac{1}{N} \sum_{i=1}^{N}(\hat{y_i}-y_i)" /></a>

<br>

**To optimize the slope for each dimension (*d*) of the input data, iterate this equation:** <br><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)x_i" title="\theta_0(t) = \theta_0(t-1) - \alpha \frac{1}{N} \sum_{i=1}^{N}(\hat{y_i}-y_i)x_i" /></a>

## 3. Results

![](https://i.imgur.com/ZrYHIVq.png)


![](https://i.imgur.com/vr20zSQ.png)
