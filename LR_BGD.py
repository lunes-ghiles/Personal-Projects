import numpy as np
import pandas as pd


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_iter=10000):
        """
        Initialize the model with learning rate and  max number of iterations.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the model using batch gradient descent. 
        Dataset: N data points, d features
        
        Parameters:
        X: numpy array, shape (N, d)
           Features
        y: numpy array, shape (N,)
           Binary target values (0 or 1).
        """
        parameters = np.ones((X.shape[1],1), dtype = np.float32)
        for i in range(self.max_iter):
            N = X.shape[0]
            delta = (-1/N) * (X.T @ (y - self.sigmoid(X @ parameters))) 
            parameters = parameters - self.learning_rate * delta
        self.parameters = parameters
        # Note I used the following as a reference:
        # Speech and Language Processing. Daniel Jurafsky & James H. Martin Chapter 5

        

    def predict_prob(self, X):
        """
        Predict probability estimates for input data X.
        """
        return self.sigmoid(X @ self.parameters)

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels (0 or 1) for input data X using a threshold.
        """
        return np.where(self.predict_prob(X) >= threshold, 1, 0)


# Example usage:
if __name__ == "__main__":
    
    # load dataframe using pandas from the .csv file
    df = pd.read_csv('loan_application.csv')

    X = df[['annual_income', 'credit_score']]
    y = y = df['loan_approved']
    
    # turn into numpy arrays
    X = X.values
    y = y.values
    
    # Feature scaling
    X = (X - np.mean(X))/np.std(X)

    # Padding for the bias term here rather than repeating in every function
    X = np.pad(X,[(0,0),(1,0)], mode = 'constant', constant_values = 1)

    
    # Create the model and train it
    model = LogisticRegressionClassifier(learning_rate=0.1, max_iter=10)
    model.fit(X, y)
    
    # Predict using the training set and get the training accuracy 
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
