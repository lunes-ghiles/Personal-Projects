
import numpy as np


class PolynomialRegression:
    def __init__(self, K, l2_coeff=0):
        """ Initialize the class of the polynomial regression model.
            For given a single scalar input x:
            f(x) = b + w_1 * x + w_2 * x^2 + ... + w_K * x^K

        args:
            - K (int): The degree of the polynomial. Note: 1 <= K <= 15
            - parameters (ndarray (shape: (K + 1, 1))): The model parameters.
            - l2_coeff (float): The coefficient of the L2 regularization term.

        NOTE: The bias term is the first element in self.parameters
            (i.e. self.parameters = [b, w_1, ..., w_K]^T).
        """
        assert 1 <= K <= 15, f"K must be between 1 and 15. Got: {K}"
        assert l2_coeff >= 0, f"l2_coeff must be non-negative. Got: {l2_coeff}"

        self.K = K
        self.l2_coeff = l2_coeff
        self.parameters = np.ones((K + 1, 1), dtype=np.float32)

    def predict(self, X):
        """ This method evaluates the polynomial model at N input data points.
        You need to use self.parameters and the input X

        args:
            - X (ndarray (shape: (N, 1))): A column vector consisting N scalar input data.
        output:
            - prediction (ndarray (shape: (N, 1))): A column vector consisting N scalar output data.

        NOTE: You MUST NOT iterate through inputs.
        """
        assert X.shape == (X.shape[0], 1)

        # ====================================================
        #  return the output of the polynomial function     # Note that we take into account padding using x^0 to fill first column
        X = np.vander(X.flatten(), self.K + 1, increasing=True)  # Set up our vandermonde matrix for the polynomials
        prediction = X @ self.parameters                         # Matrix multiply Kx1 by NxK
        # ====================================================
        return prediction

    def fit(self, train_X, train_Y):
        """ This method fits the model parameters, given the training inputs and outputs.
            This method does not have output. You only need to update self.parameters.

        args:
            - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
            - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.

        NOTE: Review from notes the least squares solution.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0],
                                                                    1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."
        assert train_X.shape[
                   0] >= self.K, f"require more data points to fit a polynomial (train_X: {train_X.shape}, K: {self.K}). Do you know why?"

        # ===========================================================================
        # Set self.parameters to least square solution for the polynomial basis
        X = np.vander(train_X.flatten(), self.K + 1, increasing=True)
        p1 = np.linalg.pinv(X.T @ X)  # (X^T X) inverse
        p2 = X.T @ train_Y            # X^T Y
        self.parameters = p1 @ p2
        # ===========================================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def fit_with_l2_regularization(self, train_X, train_Y):
        """ This method fits the model parameters with L2 regularization, given the training inputs and outputs.
        This method does not have output. You only need to update self.parameters.

        args:
            - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
            - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.

        NOTE: Review from notes the least squares solution when l2 regularization is added.
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0],
                                                                    1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."

        # =======================================================================================
        # Set self.parameters to regularized least square solution for the polynomial basis
        X = np.vander(train_X.flatten(), self.K + 1, increasing=True)
        I = np.eye(self.K + 1)
        I[0,0] = 0
        p1 = np.linalg.pinv(X.T @ X + (self.l2_coeff * I))  # (X^T X - \lambda I) inverse
        p2 = X.T @ train_Y  # X^T Y
        self.parameters = p1 @ p2
        # =======================================================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def compute_mse(self, X, observed_Y):
        """ This method computes the mean squared error.

        args:
            - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar inputs.
            - observed_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        output:
            - mse (float): The mean squared error between the predicted Y and the observed Y.
        """
        assert X.shape == observed_Y.shape and X.shape == (
            X.shape[0], 1), f"input and/or output has incorrect shape (X: {X.shape}, observed_Y: {observed_Y.shape})."

        # ========================================================================================
        # Run prediction over X and compute the mean squared error with the observed outputs
        y_hat = np.vander(X.flatten(), self.K + 1, increasing=True) @ self.parameters
        error = observed_Y - y_hat
        mse = (np.linalg.norm(error) ** 2) / observed_Y.shape[0]
        # ========================================================================================

        return mse

if __name__ == "__main__":
    model = PolynomialRegression(K=1)
    train_X = np.expand_dims(np.arange(10), axis=1)
    train_Y = np.expand_dims(np.arange(10), axis=1)

    model.fit(train_X, train_Y)
    optimal_parameters = np.array([[0.], [1.]])
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Change parameters to suboptimal for next test
    model.parameters += 1
    model.fit_with_l2_regularization(train_X, train_Y)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    # Regularization pulls the weights closer to 0.
    model = PolynomialRegression(K=1, l2_coeff=0.5)
    optimal_parameters = np.array([[0.02710843], [0.9939759]])
    model.fit_with_l2_regularization(train_X, train_Y)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))