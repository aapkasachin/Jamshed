import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from scipy.stats import mode 
from sklearn.neighbors import KNeighborsClassifier 

class K_Nearest_Neighbors_Classifier() :  
    def __init__( self, K ) : 
        self.K = K 

    def fit( self, X_train, Y_train ) : 
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.m, self.n = X_train.shape 

    def predict( self, X_test ) : 
        self.X_test = X_test 
        self.m_test, self.n = X_test.shape 
        Y_predict = np.zeros( self.m_test ) 

        for i in range( self.m_test ) : 
            x = self.X_test[i] 
            neighbors = np.zeros( self.K ) 
            neighbors = self.find_neighbors( x ) 
            Y_predict[i] = mode( neighbors )[0][0]     

        return Y_predict 

    def find_neighbors( self, x ) : 
        euclidean_distances = np.zeros( self.m ) 

        for i in range( self.m ) :
            d = self.euclidean( x, self.X_train[i] ) 
            euclidean_distances[i] = d 

        inds = euclidean_distances.argsort() 
        Y_train_sorted = self.Y_train[inds] 
        return Y_train_sorted[:self.K] 

    def euclidean( self, x, x_train ) : 
        return np.sqrt( np.sum( np.square( x - x_train ) ) ) 


def main() : 

    df = pd.read_csv( "diabetes.csv" ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values 

    X_train, X_test, Y_train, Y_test = train_test_split(  
      X, Y, test_size = 1/3, random_state = 0 ) 

    model = K_Nearest_Neighbors_Classifier( K = 3 ) 
    model.fit( X_train, Y_train ) 

    Y_pred = model.predict( X_test ) 
    correctly_classified = 0

    count = 0
    for count in range( np.size( Y_pred ) ) : 
        if Y_test[count] == Y_pred[count] : 
            correctly_classified = correctly_classified + 1
        count = count + 1

    print( "Accuracy on test set by our model       :  ", (  
      correctly_classified / count ) * 100 ) 
      
if __name__ == "__main__" :  
    main()
