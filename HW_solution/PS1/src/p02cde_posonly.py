import numpy as np
from pathlib import Path
import util
from p01fgh_plot import plot_decision_boundary, PLOT_DIR
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # load training dataset, note that the true rows in the last column indicates true in t
    x_train, t_train = util.load_dataset(train_path, label_col = "t", add_intercept = True) # syntax : extract the true examples(with a true y)
    model_c = LogisticRegression()
    model_c.fit(x_train, t_train)

    # load test dataset
    x_test, t_test = util.load_dataset(test_path, label_col = "t", add_intercept = True)
    pred_c = model_c.predict(x_test)

    # write into the pred_path
    np.savetxt(pred_path_c, pred_c)

    # plot decision boundary for (c)
    plot_decision_boundary(
        x_test[:, 1:],
        t_test,
        model_c.theta,
        title = "Problem 2(c): True labels",
        save_path = PLOT_DIR / "p02c_boundary.png"
    )

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train_d, y_train_d = util.load_dataset(train_path, label_col = "y",  add_intercept = True)
    model_d = LogisticRegression()
    model_d.fit(x_train_d, y_train_d)

    # test dataset
    x_test_d, y_test_d = util.load_dataset(test_path, label_col = "y", add_intercept = True)
    pred_d = model_d.predict(x_test_d)
    np.savetxt(pred_path_d, pred_d)

    # plot decision boundary for (d)
    plot_decision_boundary(
        x_test[:, 1:],
        t_test,
        model_d.theta,
        title="Problem 2(d): Observed Labels",
        save_path=PLOT_DIR / "p02d_boundary.png"
    )

    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid_e, y_valid_e = util.load_dataset(valid_path, label_col = "y", add_intercept = True)
    positive_mask = (y_valid_e == 1)
    alpha = np.mean(model_d.predict(x_valid_e)[positive_mask])
    print(f"Estimated alpha = {alpha:.4f}")
    # Apply correction factor 
    pred_e = pred_d / alpha
    np.savetxt(pred_path_e, pred_e)

    # Plot and use np.savetxt to save outputs to pred_path_e
    theta_e = model_d.theta.copy()

    # Shift the intercept to account for alpha
    theta_e[0] -= np.log(alpha / (2 - alpha))

    plot_decision_boundary(
        x_test[:, 1:],
        t_test,
        theta_e,
        title="Problem 2(e): Corrected Predictions",
        save_path=PLOT_DIR / "p02e_boundary.png"
    )
    # *** END CODER HERE

if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent
    data_dir = src_dir.parent / "data"
    main(
        train_path=str(data_dir / "ds3_train.csv"),
        valid_path=str(data_dir / "ds3_valid.csv"),
        test_path=str(data_dir / "ds3_test.csv"),
        pred_path=str(src_dir / "output" / "p02X_pred.txt")
    )
    print("Saved p02c_pred.txt, p02d_pred.txt, p02e_pred.txt to {}".format(src_dir / "output"))
