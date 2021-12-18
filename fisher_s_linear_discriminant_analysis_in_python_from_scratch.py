"""
    Fischer's Linear Discriminant Anlysis in Python from scratch
    Eigen -> SVD solver
    # https://sebastianraschka.com/Articles/2014_python_lda.html
    # https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd
"""

import numpy as np
from numpy.linalg.linalg import eigvals
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LDA:
    def __init__(self, project_num_dim):
        self.project_num_dim = project_num_dim
        self.labels = [0, 1] # total label
        self.dim = 0
        self.nk = None
        
    def fit(self, X, Y):
        self.dim = 2 # column dimension
        self.calculate_means(X, y)
        self.calculate_SB_SW(X)
        self.calculare_eigen_values()
    
    def calculate_means(self, X, y):
        self.class_mean = [] # each column per class 
        # [ [mean_col1_label1, mean_col2_label1], [mean_col1_label2, mean_col2_label2] ]
        self.overall_mean = np.mean(X, axis=0) # each column all class
        self.nk = []
        for label in self.labels:
            label_idx = np.where(y == label)
            _class_mean = X[label_idx].mean(axis=0) # mean each column
            self.class_mean.append(_class_mean)
            self.nk.append(X[label_idx].shape[0])
        self.class_mean = np.array(self.class_mean)
        # class_mean {0: array([-0.02778514, -1.00130985]), 1: array([0.03235729, 0.98284059])}
        # overall_mean [ 0.00228607 -0.00923463]

    
    def calculate_SB_SW(self, X):
        # calculate self.SB
        self.SB = np.zeros((self.dim, self.dim))
        # ยังติดเรื่องของ transponse
        for i in range(self.class_mean.shape[0]): # e
            mk_minus_m = np.array([self.class_mean[i]-self.overall_mean])
            _nk = self.nk[i]
            mk_minus_m_t = mk_minus_m.transpose()
            temp = (mk_minus_m_t*_nk).dot(mk_minus_m)
            self.SB += temp
        # [[9.04278115e-01 2.98329097e+01]
        # [2.98329097e+01 9.84213248e+02]]
        
        ##
        self.SW = np.zeros((self.dim, self.dim))
        for i in range(self.class_mean.shape[0]): 
            mk = self.class_mean[i] # class i
            label_idx = np.where(y == i)
            for j in label_idx[0]:
                xnk_minus_mk = np.array([X[j] - mk])
                xnk_minus_mk_t = xnk_minus_mk.transpose()
                self.SW += xnk_minus_mk_t.dot(xnk_minus_mk)
    
    def calculare_eigen_values(self):
        mat = np.dot(np.linalg.pinv(self.SW), self.SB)
        eigvals, eigvecs = np.linalg.eig(mat)
        eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
        eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)
        ## project to dimension here ตรงนี้ละคือจะเลือกว่ายัง
        w = np.array([eiglist[i][1] for i in range(self.project_num_dim)])
        # [[-0.03304637 -0.99945382]]
        self.w = w
    
    def transform_data(self, X):
        _return_x = None
        for i in range(X.shape[0]):
            _tf_data = self.w.dot(X[i]) # mean each column
            ##
            if _return_x is None:
                _return_x = _tf_data.copy()
            else:
                if _tf_data.ndim == 2:
                    _return_x = np.vstack((_return_x, _tf_data))
                else:
                    _return_x = np.append(_return_x, _tf_data)
        return _return_x
        
if __name__=='__main__':
    data = pd.read_csv('./lda-data-test1.csv')
    X = data[['attr1', 'attr2']].values
    y = data['label'].values

    _lda = LDA(1)
    _lda.fit(X, y)
    X_prime = _lda.transform_data(X)
    x_df = pd.DataFrame({
        'label': y,
        'x_prime': X_prime
    })
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="attr1", y="attr2", hue="label")
    # sns.histplot(data=x_df, x="x_prime", hue="label")
    plt.show()