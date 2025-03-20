
import numpy as np

class RBFRegression():
    def __init__(self, centers, widths):
        """ This class represents a radial basis function regression model.
        Given a single scalar input x,
        f(x) = b + w_1 * b_1(x) + w_2 * b_2(x) + ... + w_K * b_K(x), 
        where b_i is the i'th radial basis function.

        args:
            - centers (ndarray (Shape: (K, 2))): 
                A Kx2 matrix corresponding to the centers of the 2D radial basis functions.
            - widths (ndarray (Shape: (K, 1))): 
                A K-column vector corresponding to the widths of the radial basis functions.
        """
        assert centers.shape[0] == widths.shape[0], f"The number of centers and widths must match. (Centers: {centers.shape[0]}, Widths: {widths.shape[0]})"
        assert centers.shape[1] == 2, f"Each center should have two components. (Centers: {centers.shape[1]})"
        assert widths.shape[1] == 1, f"Each width should have one component. (Widths: {widths.shape[1]})"
        self.centers = centers
        self.widths = widths
        self.K = centers.shape[0]

        # Remember that we have K weights and 1 bias.
        self.parameters = np.ones((self.K + 1, 1), dtype=np.float32)

    def rbf_2d(self, X, i):
        """ This method computes the output of the i'th 2D radial basis function given the inputs.
        Recall that RBF(x) = exp(-||x - center||^2 / (2 * width^2))

        args:
            - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.
            - i (int): The i'th radial basis function.

        output:
            - z ndarray (Shape: (N, 1)): A N-column vector consisting N scalar output data.
        """
        assert 1 <= i <= self.K

        # Retrieve the center and the width of the radial basis function
        rbf_center = self.centers[[i-1]]
        rbf_width = self.widths[[i-1]]

        # ====================================================
        # Use the selected center and width to compute the RBF
        p1 = X - rbf_center
        numerator = -((p1 ** 2).sum(axis = 1)) # Maybe not the most efficient implementation, but works!
        rbf_out = np.exp(numerator/(2*(rbf_width ** 2))).T
        # ====================================================
        return rbf_out

    def predict(self, X):
        """ This method predicts the output of the given input data using the model parameters. 

        args:
            - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.

        output:
            - ndarray (shape: (N, 1)): A N-column vector consisting N scalar output data.

        NOTE: You must not iterate through inputs. HINT: You can use self.rbf_2d to compute b_i(X).
        """
        assert X.shape[1] == 2, f"Each input should contain two components. Got: {X.shape[1]}"

        # ==================================================================================================
        # Use rbf_2d function to compute each radial basis function at X inputs and then sum them all. 
        arrays = [self.rbf_2d(X,i) for i in range(1, self.K + 1)]
        B = np.hstack(arrays)  # This gives us the transpose...might as well go with it
        aug_B = np.pad(B, [(0,0),(1,0)], mode='constant', constant_values=1)
        prediction = aug_B @ self.parameters
        # ==================================================================================================
        return prediction
    
    def fit_with_l2_regularization(self, train_X, train_Y, l2_coeff):
        """ This method fits the model parameters, given the training inputs and outputs.
        This method does not have output. You only need to update self.parameters.

        args:
            - train_X (ndarray (shape: (N, 2))): A Nx2 matrix consisting N 2D training inputs.
            - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
            - l2_coeff (float): The lambda term that decides how much regularization we want.

        NOTE: Review from notes the least squares solution with l2 regularization.
        """
        assert train_X.shape[0] == train_Y.shape[0], f"Number of inputs and outputs are different. (train_X: {train_X.shape[0]}, train_Y: {train_Y.shape[0]})"
        assert train_X.shape[1] == 2, f"Each input should contain two components. Got: {train_X.shape[1]}"
        assert train_Y.shape[1] == 1, f"Each output should contain 1 component. Got: {train_Y.shape[1]}"

        # ====================================================
        # Set self.parameters to regularized least square solution for the radial basis function
        arrays = [self.rbf_2d(train_X,i) for i in range(1, self.K + 1)]
        B = np.hstack(arrays)
        aug_B = np.pad(B, [(0,0),(1,0)], mode='constant', constant_values=1)
        I = np.eye(self.K + 1)
        I[0,0] = 0
        p1 = np.linalg.pinv(aug_B.T @ aug_B + (l2_coeff * I)) # (X^T X - \lambda I) inverse
        p2 = aug_B.T @ train_Y  # X^T Y
        self.parameters = p1 @ p2
        # ====================================================

        assert self.parameters.shape == (self.K + 1, 1)


if __name__ == "__main__":
    # NOTE: This is just a quick check but does not cover all cases.
    centers = np.tile(np.expand_dims(np.arange(2), axis=1), reps=(1, 2))
    widths = np.ones((2, 1))
    model = RBFRegression(centers, widths)

    train_X = np.tile(np.expand_dims(np.arange(3), 1), reps=(1, 2))
    train_Y = np.array([[4.10363832], [4.73575888], [2.1402696]])

    optimal_parameters = np.array([[1], [2], [3]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coeff=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[2.70393756], [0.87280156], [0.95519078]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coeff=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))