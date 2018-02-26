# Gaussian Naive Bayes for Income Data Set
GNB classifier trained to predict individual's income

## Data Set
Data set used in this implementation is [Adult Data](https://archive.ics.uci.edu/ml/datasets/adult) set of UCI machine learning, visit the website for more information

## Implementation detail
* Ignore any unknwon attributes that marked as "?" in the dataset and sum up the probability of the occurance of discrete attribute to 1
* Assume the result of log 0 is negative infinite.

## Training Result
The prediction error obtained from this implementation is 16.90%, which is larger
than the result descripted on the website, which is 16.12%. Probably it is because I didn't use smoothing in this implementation

## Libaries
* [Numpy]
