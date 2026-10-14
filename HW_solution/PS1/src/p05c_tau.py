import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression
from p05b_lwr import main as evaluate_lwr


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_test, y_test = util.load_dataset(test_path, add_intercept = True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    minMse =  evaluate_lwr(tau_values[0], train_path, valid_path, plot_name="p05c")
    bestTau = tau_values[0]
    
    for tau in tau_values[1:]:
        mseCur = evaluate_lwr(tau, train_path, valid_path, plot_name="p05c")
        
        if mseCur < minMse:
            minMse = mseCur
            bestTau = tau

    print(f"bestTau: {bestTau}")

    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(bestTau)
    model.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    MSENew = evaluate_lwr(bestTau, train_path, test_path, plot_name="p05c_test")
    print(f"the newest MSE with bestTau{bestTau} is: {MSENew}")
    
    predictions = model.predict(x_test)
    # Save predictions to pred_path
    np.savetxt(pred_path, predictions)
    
    # We already have plotted six figures when extracting bestTau value, and the last one is
    # about the single model with the best_tau value, and the new test set
    
    # *** END CODE HERE ***


if __name__ == "__main__":
    main(
        tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='../data/ds5_train.csv',
         valid_path='../data/ds5_valid.csv',
         test_path='../data/ds5_test.csv',
         pred_path='output/p05c_pred.txt'
    )