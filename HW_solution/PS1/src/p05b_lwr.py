import numpy as np
import util

from linear_model import LinearModel
from plot_lwr import plot_lwr

def MSE(model, train_path, eval_path):
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    return (mse,y_pred)



def main(tau, train_path, eval_path, plot_name="p05b"):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        plot_name: Prefix for the saved plot filename.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)

    # *** START CODE HERE ***
    model = LocallyWeightedLinearRegression(tau)
    mse, y_pred = MSE(model, train_path, eval_path)
    plot_lwr(x_train, y_train, x_eval, y_pred, tau, name=plot_name)
    print(f"mse:{mse}")
    return mse
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _ = x.shape
         
        # predictions is a list of numbers, in which each row represents 
        # a predicted score specifically for example i
        predictions = np.zeros(m)
        for i, query in enumerate(x):
            # we want to derive the weight matrix
            # broadCasting: (m,n) - (n,) -> (m,n)
            differences = self.x - query
            # ||x^{i} - x||^2
            squared_distance = np.sum(differences ** 2, axis = 1)
            # w^{i} = exp{-|x^{i} - x||^2 /2 \tau^2 }
            # note the dim(squared_distance is (m, ))
            weights = np.exp(-squared_distance / (2 * self.tau ** 2) )
            # we didn't use keepdim = True, hence the dimension here is just (m,)

            # Note that we can also use keepdims as a shorthand for preserving the dimensions
            # squared_distances = np.sum(differences ** 2, axis = 1, keepdims = True)
            # weights = np.exp(-squared_distance / (2 * self.tau ** 2) )
            # A = self.x.T @ (weights * self.x)
            # Now use the normal equation
            # Recall that X^{T}WX^{-1} * X^{T}Wy
            # Syntax: @ is matrix multiplication, * is matrix Hadamard product(elementWise prod)
            # Dim analysis : x(m,n); weights[:,None] -> weights(m,1)
            A = self.x.T @ (weights[:,None] * self.x)
            b = self.x.T @ (weights * self.y)
            try: 
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(A) @ b

            predictions[i] = query @ theta

        return predictions
    
        # Remark: note that for LWR, we don't just do X^T @ theta for the whole valid examples.
        # By nature of W depending on x^{i} - x, we need to update W and theta* for each differente validation example, 
        # and store them in a predictions array

    
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(
        tau=0.5,
        train_path="../data/ds5_train.csv",
        eval_path="../data/ds5_valid.csv"
    )