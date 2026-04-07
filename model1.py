import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib


def main():
    lr = LinearRegression()
    mse_scores = []

    # Repeat experiment 10 times and keep final trained model
    for seed in range(10):
        rng = np.random.RandomState(seed)
        x = 10 * rng.rand(1000).reshape(-1, 1)
        y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=50
        )

        lr.fit(X_train, y_train)
        y_preds = lr.predict(X_test)

        test_mse = mean_squared_error(y_test, y_preds)
        mse_scores.append(test_mse)
        print(f"Run {seed + 1} MSE: {test_mse}")

    average_mse = float(np.mean(mse_scores))
    print("Average Mean Squared Error:", average_mse)

    # Save trained model
    joblib.dump(lr, "model1.pkl")

    # Write metrics
    with open("metrics.txt", "w", encoding="utf-8") as outfile:
        outfile.write(f"Mean Squared Error = {average_mse}\n")

    # Plotting using the final train/test split and predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, label="Training data")
    plt.scatter(X_test, y_test, label="Testing data")
    plt.scatter(X_test, y_preds, label="Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Training and Testing Data Split")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model_results.png", dpi=120)
    print("Files generated: model1.pkl, metrics.txt, model_results.png")


if __name__ == "__main__":
    main()
