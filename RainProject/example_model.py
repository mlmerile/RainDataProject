import data_util as du
import eval_crps

import numpy as np
## read 10000 lines
data = du.read_raincsv("Data/train_2013.csv",10000)
(n_sample,n_feature) = data.shape

## Split data
rows_train = np.random.choice(data.index,n_sample//2)
data_train = data.ix[rows_train]
data_test = data.drop(rows_train)

mean_expected = np.mean(data_train["Expected"])

(n_sample_test,n_feature_test) = data_test.shape
res = np.zeros((n_sample_test,70))

### STRAT 1
res[:,round(mean_expected):] = 1

## Compute the score
score = eval_crps.crps(res,data_test["Expected"].values)

### STRAT 2
hist,bins = np.histogram(data_train["Expected"].values,bins=np.insert(np.insert(np.arange(1.0,70,1),0,0.000001),0,0))
proba = np.cumsum(hist)
proba = proba.astype(float)
proba = proba / proba[-1]
res = np.array([proba,]*n_sample_test)
score2 = eval_crps.crps(res,data_test["Expected"].values)

score3 = eval_crps.crps(np.ones((n_sample_test,70)),data_test["Expected"].values)

## Test if it is better to be good on bad data or not
res = np.ones((n_sample_test,70))
select_bad = (data_test["Expected"] > 70).values
res[select_bad,:] = np.zeros(70)
score4 = eval_crps.crps(res,data_test["Expected"].values)

res = np.ones((n_sample_test,70))
greater_zero = (data_test["Expected"] > 0).values
less_one = (data_test["Expected"] < 1).values
select_bad = np.logical_and(greater_zero,less_one)
res[select_bad,:] = np.array([0] + [1]*69)
score5 = eval_crps.crps(res,data_test["Expected"].values)

