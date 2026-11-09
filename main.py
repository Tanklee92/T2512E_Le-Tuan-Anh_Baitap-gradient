import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

n_samples = 100

X = np.random.uniform(1.0, 10.0, n_samples)

y_real = 5.0 + 2.5 * X + np.random.normal(0, 1.5, n_samples)

print("Kich thuoc X:", X.shape)
print("Kich thuoc y:", y_real.shape)

print("5 chi phi quang cao dau tien:", X[:5].round(2))
print("5 doanh so dau tien:", y_real[:5].round(2))


def compute_loss(X, y, w, b):

    y_pred = w * X + b

    mse = np.mean((y - y_pred) ** 2)

    return mse


def compute_gradients(X, y, w, b):

    N = len(X)

    y_pred = w * X + b

    error = y - y_pred

    dw = (-2 / N) * np.sum(X * error)

    db = (-2 / N) * np.sum(error)

    return dw, db


w = 0.0
b = 0.0

loss = compute_loss(X, y_real, w, b)

dw, db = compute_gradients(X, y_real, w, b)

print("\n===== KET QUA TASK 1 =====")
print(f"Loss ban dau: {loss:.4f}")
print(f"dw = {dw:.4f}")
print(f"db = {db:.4f}")



w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000

history_loss = []

print("\n===== KET QUA TASK 2 =====")

for epoch in range(1, epochs + 1):

    dw, db = compute_gradients(X, y_real, w, b)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss = compute_loss(X, y_real, w, b)

    history_loss.append(loss)

    if epoch in [1, 200, 400, 600, 800, 1000]:
        print(
            f"Epoch {epoch}: "
            f"w = {w:.4f}, "
            f"b = {b:.4f}, "
            f"Loss = {loss:.4f}"
        )



os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    range(1, epochs + 1),
    history_loss,
    color="blue"
)

axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(True)

axes[1].scatter(
    X,
    y_real,
    color="blue",
    label="Actual Data"
)

X_line = np.linspace(X.min(), X.max(), 100)
y_line = w * X_line + b

axes[1].plot(
    X_line,
    y_line,
    color="red",
    label="Regression Line"
)

axes[1].set_title("Linear Regression")
axes[1].set_xlabel("Advertising Cost")
axes[1].set_ylabel("Sales")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

plt.savefig("results/task3_training.png", dpi=150)

plt.show()


def run_experiment(lr, epochs=200):

    w = 0.0
    b = 0.0

    losses = []

    for epoch in range(epochs):

        dw, db = compute_gradients(X, y_real, w, b)

        w = w - lr * dw
        b = b - lr * db

        with np.errstate(over="ignore", invalid="ignore"):
            loss = compute_loss(X, y_real, w, b)

        losses.append(loss)

    return w, b, losses


learning_rates = [0.0001, 0.01, 0.15]

print("\n===== KET QUA TASK 4 =====")

for lr in learning_rates:

    w_test, b_test, losses = run_experiment(lr)

    print(f"\nLearning Rate = {lr}")
    print(f"w = {w_test:.4f}")
    print(f"b = {b_test:.4f}")
    print(f"Loss cuoi = {losses[-1]:.4f}")

