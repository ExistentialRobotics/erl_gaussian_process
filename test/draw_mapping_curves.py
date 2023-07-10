import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-10, 10, 200)
    ys = {
        # "identity": x,
        # "inverse": 1 / x,
        # "exp": np.exp(-x),
        "tanh": np.tanh(x),
        "sigmoid": 1 / (1 + np.exp(-x)),
        "sigmoid_-2": 1 / (1 + np.exp(2 * x)),
        "sigmoid_2": 1 / (1 + np.exp(-2 * x)),
        "sigmoid_-0.5": 1 / (1 + np.exp(0.5 * x)),
        "sigmoid_0.5": 1 / (1 + np.exp(-0.5 * x)),
        "sigmoid_-0.25": 1 / (1 + np.exp(0.25 * x)),
        "sigmoid_0.25": 1 / (1 + np.exp(-0.25 * x)),
    }

    for name, y in ys.items():
        plt.plot(x, y, label=name)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
