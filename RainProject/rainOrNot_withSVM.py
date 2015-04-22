import data_util as du
import numpy as np
import matplotlib.pyplot as plt

## read 10000 lines
data = du.read_raincsv("Data/train_2013.csv",100)
(n_sample,n_feature) = data.shape

##Create features
X = du.dataFrameToMatrix( \
du.dataToHist( data, ["RR1","MassWeightedMean","Composite"],du.build_col_to_quantilehist(5)) )

##Remove nan values
X[ np.isnan(X) ] = 0.0

##Class
Y = np.vstack(data.Expected)
Y[Y>0.0] = 1

##define kernel function
distance_matrix = np.ones((50,50)) - np.identity(50)
def emdKernel(X,Y,distance_matrix = distance_matrix ) : 
    return emd(X,Y, distance_matrix)


##Pca plot
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2,kernel=emdKernel).fit(X)
reduced_X = pca.transform(X)
plt.scatter(reduced_X[Y[:,0]==0, 0], reduced_X[Y[:,0]==0, 1],color='blue')
plt.scatter(reduced_X[Y[:,0]!=0, 0], reduced_X[Y[:,0]!=0, 1],color='red')
plt.show()

##Svm
from sklearn import svm
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel=emdKernel, C=C).fit(X, Y)

svc.score(X,Y)
predicted_Y = svc.predict(X)
<<<<<<< HEAD

 from pyemd import emd 
 first_signature = np.array([0.0, 1.0])
 second_signature = np.array([5.0, 3.0])
distance_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
emd(first_signature, second_signature,distance_matrix)
=======
>>>>>>> 3b7f5fa3eb29690f2354f35e4d013440ecfe4e90
