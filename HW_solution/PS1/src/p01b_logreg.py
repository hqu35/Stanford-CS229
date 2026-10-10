import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # extractbthe eval path, calculate the predictions, as well as writing the predictions into the destined file path
    x_evaluation, _ = util.load_dataset(eval_path, add_intercept = True)
    predictions = model.predict(x_evaluation)
    np.savetxt(pred_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).(i.e. X)
            y: Training example labels. Shape (m,). (i.e. Y)
        """
        # *** START CODE HERE ***
        # Newton's algorithm: x_{k+1} = x_k - (H(x_k))^{-1} \nabla J(x_k)
        
        m,n = x.shape

        #initialize theta to 0, and epsilon
        self.data = np.zeros(n)
        while True:
            # predictions
            z = x @ self.theta
            h = 1 / (1 + np.exp(-np.clip(z, -500, 500))) # g(theta^T * X)

            # first order gradient 
            gradient = (x.T @ (h - y)) / m

            # Hessian
            weights = h * (1-h) # 1D vector of size (m)
            # note that this is just H = 1/m * (XWX^T) 
            # also note that weights in Numpy is of Dim 1D vector, weights[:,None]
            # change weights from 1D vector to a 2D vector i.e. (m,1)
            # Dimension check (n,m) * (m,m) * (m, n)
            H = (x.T @ (weights[:, None] * x)) / m

            # note, sometimes broadcasting is hard to achieve, we can call the diag method
            W = np.diag(weights)
            H2 = (x.T @ W * x) / m

            assert(np.allclose(H, H2))

            # Newton's update
            # Note that np.linalg.solve(A,b) returns x = A^{-1}b
            delta = np.linalg.solve(H, gradient)
            theta_new = self.theta - delta

            # Lastly, check error upper bound
            # note: this is just euclidean norm d(x,y) ^ ord
            if np.linalg.norm(theta_new - self.theta, ord = 1) < epsilon:
                self.theta = theta_new
                break
            self.theta = theta_new

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        predictions = 1 / 1 + np.exp(-np.clip(z,-500, 500))
        return predictions
        # *** END CODE HERE ***
