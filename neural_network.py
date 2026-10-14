import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Данные

np.random.seed(42)

digits = load_digits()
X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


def one_hot(y, n_classes=10):
    result = np.zeros((len(y), n_classes))
    result[np.arange(len(y)), y] = 1
    return result


Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

print("Обучение:", X_train.shape, " Тест:", X_test.shape)


# Функция активации, softmax, кросс-энтропия

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))


def accuracy(probs, y):
    return np.mean(np.argmax(probs, axis=1) == y)


# Архитектура сети и начальные веса

n_input, n_hidden, n_output = 64, 32, 10

W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2 / n_input)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2 / n_hidden)
b2 = np.zeros(n_output)


# Прямой проход

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    return z1, a1, probs


# Обучение

epochs = 50
learning_rate = 0.05
batch_size = 32

history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}


def evaluate():
    _, _, p_train = forward(X_train)
    _, _, p_test = forward(X_test)
    history["train_loss"].append(cross_entropy(p_train, Y_train))
    history["test_loss"].append(cross_entropy(p_test, Y_test))
    history["train_acc"].append(accuracy(p_train, y_train))
    history["test_acc"].append(accuracy(p_test, y_test))


evaluate()

for epoch in range(1, epochs + 1):
    order = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), batch_size):
        idx = order[start:start + batch_size]
        X_batch, Y_batch = X_train[idx], Y_train[idx]

        z1, a1, probs = forward(X_batch)

        dz2 = (probs - Y_batch) / len(X_batch)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ W2.T) * relu_derivative(z1)
        dW1 = X_batch.T @ dz1
        db1 = dz1.sum(axis=0)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    evaluate()
    if epoch % 10 == 0:
        print(f"Эпоха {epoch:2d}: loss обуч. {history['train_loss'][-1]:.3f}, "
              f"loss тест {history['test_loss'][-1]:.3f}, "
              f"accuracy обуч. {history['train_acc'][-1]:.3f}, "
              f"accuracy тест {history['test_acc'][-1]:.3f}")

print(f"Accuracy на тесте до обучения:    {history['test_acc'][0]:.1%}")
print(f"Accuracy на тесте после обучения: {history['test_acc'][-1]:.1%}")


# График loss и accuracy

epoch_axis = range(epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epoch_axis, history["train_loss"], color="#2a78d6", linewidth=2, label="Loss — обучение")
plt.plot(epoch_axis, history["test_loss"], color="#2a78d6", linewidth=2, linestyle="--", label="Loss — тест")
plt.plot(epoch_axis, history["train_acc"], color="#eb6834", linewidth=2, label="Accuracy — обучение")
plt.plot(epoch_axis, history["test_acc"], color="#eb6834", linewidth=2, linestyle="--", label="Accuracy — тест")

plt.title("Как меняются loss и accuracy во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("Значение (accuracy — доля от 0 до 1)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
