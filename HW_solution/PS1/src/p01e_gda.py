import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    x_eval, _ = util.loaddataset(eval_path, add_intercept = False)
    predictions = model.predict(x_eval)
    # store the predicted data into the pred_path
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = np.mean(y)

        # Indicator function(boolean mask)
        # The first lines create a boolean masking of y (i.e. which lines of y are true and which are false)
        # and then we extract those lines with true values and extract the lines corresponding with the true values
        mask0 = (y==0)
        mask1 = (y==1)
        x0 = x[mask0]
        x1 = x[mask1]

        mu0 = np.mean(x0, axis = 0)
        mu1 = np.mean(x1, axis = 0)

        # Estimate the shared covariance matrix
        # \sum_{i, y^{i} == 0} (x^{i} - mu0)(x^{i} - mu0)^T
        # \sum_{i, y^{i} == 1} (x^{i} - mu1)(x^{i} - mu1)^T
        residual0 = x0 - mu0
        residual1 = x1 - mu1
        # note that Batch form stacks examples transpose
        sigma = (residual0.T @ residual0  + residual1.T @ residual1)/m

        # Derive theta from the GDA params
        theta = np.linalg.solve(sigma, mu0 - mu1)
        theta0 = np.log(phi/(1 - phi)) - 0.5 * ((mu0 + mu1) @ theta)
        self.theta = np.concatenate(([theta0], theta))
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # python syntax array[begin, end, step]
        z = x @ self.theta[1:] + self.theta[0]
        predictions = 1 / 1 + np.exp(-np.clip(z,-500,500))
        # *** END CODE HERE
