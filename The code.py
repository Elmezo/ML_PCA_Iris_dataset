#Iris data set (matrix 150*4) ---> Labeled Data (R4 ---> R2)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#Load Iris dataset
iris = load_iris()
x = iris.data       # the recorded data with features
y = iris.target     # the label column
print("Feature names: ", iris.feature_names)
print("Shape of data: ", x.shape)
print("Shape of y: ", y.shape)   # Vector (150*1)
print(y)


#Step 1: Standardization
x_std = StandardScaler().fit_transform(x)
        # Z = (x - mean) / segma
        
#Step 2: Covariance (Correlation) matrix
#Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
#Step 4: Sort eigenvalues
#Step 5: Construct the projection matrix W
#Last step: Projection onto the New Feature Space
pca_model = PCA(n_components = 2)
x_pca = pca_model.fit_transform(x_std)
print("After PCA:\n")
print("Features of size: ", x_pca.shape)
print(X_pca)
