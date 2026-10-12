import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import util

from p01b_logreg import LogisticRegression
from p01e_gda import GDA


# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"

PLOT_DIR.mkdir(exist_ok=True)

def plot_decision_boundary(x, y, theta, title, save_path):
    """
    Plot training examples and the decision boundary.

    Args:
        x: Feature matrix, shape (m, 2).
        y: Binary labels, shape (m,).
        theta: Model parameters, shape (3,).
               theta = [theta0, theta1, theta2]
        title: Title of the figure.
        save_path: Output image path.
    """
    # TODO : plot the three figures instructed in f,g,h

    fig, ax = plt.subplots(figsize=(8, 6))

    # ----------------------------------------
    # Step 1: Plot training examples
    # ----------------------------------------

    negative_mask = (y == 0)
    positive_mask = (y == 1)

    ax.scatter(
        # this is extracting the class 0 examples, and then extract the first entry
        x[negative_mask, 0],
        # Extract the second feature (x2) of all class-0 examples
        x[negative_mask, 1],
        # note that a single x1 from x[neg,0] pairs up with x2 from x[neg,1]
        marker="o",
        label="Class 0"
    )

    ax.scatter(
        x[positive_mask, 0],
        x[positive_mask, 1],
        marker="x",
        label="Class 1"
    )
    # Remark: Notice that the training examples are plotted here, the mesh descrived below has nothing
    # to do with the plotting of the training examples. 

    # ----------------------------------------
    # Step 2: Construct coordinate grid
    # ----------------------------------------

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

    pad1 = 0.1 * (x1_max - x1_min)
    pad2 = 0.1 * (x2_max - x2_min)

    x1_values = np.linspace(
        x1_min - pad1,
        x1_max + pad1,
        250
    )

    x2_values = np.linspace(
        x2_min - pad2,
        x2_max + pad2,
        250
    )

    X1, X2 = np.meshgrid(x1_values, x2_values)

    # ----------------------------------------
    # Step 3: Evaluate linear decision function
    # ----------------------------------------

    Z = (
        theta[0]
        + theta[1] * X1
        + theta[2] * X2
    )

    # ----------------------------------------
    # Step 4: Plot the zero-level contour
    # ----------------------------------------

    ax.contour(
        X1,
        X2,
        Z,
        levels=[0],
        colors="red",
        linewidths=2
    )

    # alternative ways to draw the boundary lines
    # notice that boundary line is just x2 = - \frac{\theta_0 + \theta_1x_1}{\theta_2}
    # x1_values = np.array([x1_min, x1_max])
    # x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]
    # ax.plot(x1_values, x2_values, color = "red", lindwidth = 2)

    # ----------------------------------------
    # Step 5: Formatting
    # ----------------------------------------

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    # Your code ends here


def run_dataset(dataset_name):
    """
    Train Logistic Regression and GDA on one dataset,
    then create their decision-boundary plots.
    """
    # TODO: train LR and GDA with the training examples, and then plot the decision boundary
    train_path = DATA_DIR / f"{dataset_name}_train.csv"

    # Original features: shape (m, 2)
    x_train, y_train = util.load_dataset(
        # str converts the path to a string
        str(train_path),
        add_intercept=False
    )

    # ----------------------------------------
    # Logistic Regression
    # ----------------------------------------

    # Add intercept column to the feature matrix.
    x_train_lr = np.column_stack(
        # Syntax: note that stack tends to stack all the columns together, 
        # in this specific case, we are adding another cols of 1 to the left of the matrix X
        (np.ones(x_train.shape[0]), x_train)
    )

    logreg = LogisticRegression()
    logreg.fit(x_train_lr, y_train)

    plot_decision_boundary(
        x_train,
        y_train,
        logreg.theta,
        title=f"{dataset_name}: Logistic Regression",
        save_path=PLOT_DIR / f"{dataset_name}_logreg.png"
    )

    # ----------------------------------------
    # Gaussian Discriminant Analysis
    # ----------------------------------------

    gda = GDA()
    gda.fit(x_train, y_train)

    plot_decision_boundary(
        x_train,
        y_train,
        gda.theta,
        title=f"{dataset_name}: GDA",
        save_path=PLOT_DIR / f"{dataset_name}_gda.png"
    )

def main():
    # Problem 1(f)
    run_dataset("ds1")

    # Problem 1(g)
    run_dataset("ds2")

    # This will generate four images 

if __name__ == "__main__":
    main()