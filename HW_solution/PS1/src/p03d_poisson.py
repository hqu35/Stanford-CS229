import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    poiModel = PoissonRegression(step_size = lr)
    poiModel.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval,_ = util.load_dataset(eval_path, add_intercept = True)
    np.savetxt(pred_path, poiModel.predict(x_eval))
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # SGD
        m, n = x.shape

        # Initialize the parameter vector
        self.theta = np.zeros(n)

        # Initialize only the intercept
        self.theta[0] = np.log(np.mean(y))
        
        max_iteration = 5000
        error = 1e-5
        for i in range(max_iteration):
            # forward pass: E[Y | X = x] = exp(X @ theta)
            predictions = np.exp(x @ self.theta)
            # the gradient of the current iteration: \nabla_theta l(theta)
            gradient = x.T @(y - predictions) / m

            # Gradient ascent
            theta_new = self.theta + self.step_size * gradient
 
            if i < 5:
                print(f"Iteration {i}")
                print("Theta:", self.theta)
                print("Gradient norm:", np.linalg.norm(gradient))
                print("Prediction range:", np.min(predictions), np.max(predictions))
                print("Updated eta range:", np.min(x @ theta_new),
                    np.max(x @ theta_new))

            # check convergence
            if np.linalg.norm (theta_new - self.theta) < error:
                self.theta = theta_new
                print(f"Converged after {i + 1} iterations.")
                break
            self.theta = theta_new
        else :
            print("Maximum number of iterations reached")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
