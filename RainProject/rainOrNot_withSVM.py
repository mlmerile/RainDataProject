import data_util as du
import numpy as np
import matplotlib.pyplot as plt
import histogram_util as hu

## read 10000 lines
data = du.read_raincsv("Data/train_2013.csv",1000)
(n_sample,n_feature) = data.shape

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

svc.score(X,Y)
predicted_Y = svc.predict(X)

