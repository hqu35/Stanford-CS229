from pathlib import Path

import matplotlib.pyplot as plt


def plot_lwr(x_train, y_train, x_eval, y_pred, tau, name="p05b"):
    """Plot LWR predictions and save to PS1/plots using the supplied prefix."""
    fig, ax = plt.subplots()
    ax.scatter(x_train[:, 1], y_train, color="blue", marker="x",
               label="Training Data")
    ax.scatter(x_eval[:, 1], y_pred, color="red", marker="o",
               label="Evaluation Predictions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Locally Weighted Regression(tau = {tau})")
    ax.legend()

    plot_dir = Path(__file__).resolve().parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{name}_lwr_tau_{tau}.png"
    fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()
    plt.close(fig)
    return plot_path
