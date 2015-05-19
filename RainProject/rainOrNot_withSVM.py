import data_util as du
import numpy as np
import matplotlib.pyplot as plt
import histogram_util as hu

## read 10000 lines
data = du.read_raincsv("Data/train_2013.csv")
(n_sample,n_feature) = data.shape
rows_train = np.random.choice(data.index,n_sample//2)
data_train = data.ix[rows_train]
data_test = data.drop(rows_train)

##Create features
X = du.dataFrameToMatrix( \
hu.dataToHist( data, ["RR1","MassWeightedMean"],hu.build_col_to_quantilehist(2)) )

##Remove nan values
X[ np.isnan(X) ] = 0.0

##Class
Y = np.vstack(data.Expected)
Y[Y>0.0] = 1


##Pca plot
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
reduced_X = pca.transform(X)
plt.scatter(reduced_X[Y[:,0]==0, 0], reduced_X[Y[:,0]==0, 1],color='blue')
plt.scatter(reduced_X[Y[:,0]!=0, 0], reduced_X[Y[:,0]!=0, 1],color='red')
plt.show()

##Svm
from sklearn import svm
C = 1.0  # SVM regularization parameter
svc = svm.SVC( C=C).fit(X, Y)

print svc.score(X,Y)

limit = 70.0

X2,Y2 = du.clean_data(data_train,x_extra=np.arange(61,-1,-10))
X2_test,Y2_test = du.clean_data(data_test,x_extra=np.arange(61,-1,-10))
Y2[Y2>limit] = 1
Y2[Y2<=limit] = 0
Y2_test[Y2_test>limit] = 1
Y2_test[Y2_test<=limit] = 0

nb_not_null_train = np.sum(Y2)
ind_sel_train = np.where(Y2<=limit)[0]
ind_sel_train_notnull = np.where(Y2>limit)[0]
nb_not_null_test = np.sum(Y2_test)
ind_sel_test = np.where(Y2_test<=limit)[0]
ind_sel_test_notnull = np.where(Y2_test>limit)[0]
rows_sel_train = np.concatenate((ind_sel_train_notnull,np.random.choice(ind_sel_train,nb_not_null_train)),axis=0)
rows_sel_test = np.concatenate((ind_sel_test_notnull,np.random.choice(ind_sel_test,nb_not_null_test)),axis=0)
X2 = X2[rows_sel_train]
Y2 = Y2[rows_sel_train]
X2_test = X2_test[rows_sel_test]
Y2_test = Y2_test[rows_sel_test]

pca2 = PCA(n_components=2).fit(X2)
reduced_X2 = pca2.transform(X2)
plt.scatter(reduced_X2[Y2==0, 0], reduced_X2[Y2==0, 1],color='blue')
plt.scatter(reduced_X2[Y2!=0, 0], reduced_X2[Y2!=0, 1],color='red')
plt.show()

C = 1000.0  # SVM regularization parameter
svc2 = svm.SVC( C=C).fit(X2, Y2)

print svc2.score(X2_test,Y2_test)



