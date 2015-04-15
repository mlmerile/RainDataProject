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
hist,bins = np.histogram(data_train["Expected"].values,bins=np.arange(71))
proba = np.cumsum(hist)
proba = proba.astype(float)
proba = proba / proba[-1]
res = np.array([proba,]*n_sample_test)
score2 = eval_crps.crps(res,data_test["Expected"].values)

score3 = eval_crps.crps(np.ones((n_sample_test,70)),data_test["Expected"].values)



