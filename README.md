# Gaussian Naive Bayes Classifier for Income Data Set
GNB classifier trained to predict individual's income

## Data Set
Data set used in this implementation is [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult) of UCI machine learning repository, visit the website for more information.

## Implementation details
* Ignore any unknwon attributes that marked as "?" in the dataset and sum up the probability of the occurance of discrete attribute to 1
* Assume the result of log 0 is negative infinite.

## Training Result
The prediction error obtained from this implementation is 16.90%, which is larger
than the result descripted on the website, which is 16.12%. Probably it is because I didn't use smoothing in this implementation.

## Libaries
* Use [Numpy](http://www.numpy.org/) for scientific computing.
