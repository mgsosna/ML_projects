# [IN PROGRESS]
# Multivariate linear regression via gradient descent

## 1. Abstract
This repository performs a linear regression via gradient descent on any-dimensional data. The arguments for `gd_lm` are as follows:
* `X`: input data
* `y`: output data
* `alpha`: the learning rate
* `n_iter`: the number of iterations to take for gradient descent
* `figure`: for 1- and 2-dimensional `X` data, should a figure be plotted?
* `stop_thresh`: when MSE improvement falls below this value, gradient descent stops
* `n_runs`: number of times to run gradient descent (each with different starting conditions)
* `full`: should the full MSE and coefficient trajectories be saved?

This repository includes the following helper functions:
* `gen_data`: generate any-dimensional random data
* `analytical_reg`: calculate the analytical solution for N-dimensional data, when N > 1 (similar to R's `lm` function, but easier for iterating in a parameter scan)
* `gen_preds`: generate model predictions on any-dimensional data, given a set of regression coefficients

[grad_desc_lm.R](grad_desc_lm.R) and [grad_desc_lm.py](grad_desc_lm.py) include all functions, and [grad_desc_demo.R](grad_desc_demo.R) includes code for visualizations.

## 2. Background
Inspired by [Andrew Ng](http://www.andrewng.org/)'s machine learning Coursera course, I decided to write a function that performs linear regression via [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).




**To optimize the model intercept, iterate this equation:**
<a href ="https://www.codecogs.com/eqnedit.php?latex=\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0(t)&space;=&space;\theta_0(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2" title="\theta_0(t) = \theta_0(t-1) - \alpha \frac{1}{N} \sum_{i=1}^{N}(\hat{y_i}-y_i)^2" /></a>

**To optimize the slope for each dimension (*d*) of the input data, iterate this equation:**
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_d(t)&space;=&space;\theta_d(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2&space;x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_d(t)&space;=&space;\theta_d(t-1)&space;-&space;\alpha&space;\frac{1}{N}&space;\sum_{i=1}^{N}(\hat{y_i}-y_i)^2&space;x_i" title="\theta_d(t) = \theta_d(t-1) - \alpha \frac{1}{N} \sum_{i=1}^{N}(\hat{y_i}-y_i)^2 x_i" /></a>

## 3. Results

![](https://i.imgur.com/ZrYHIVq.png)


![](https://i.imgur.com/vr20zSQ.png)
