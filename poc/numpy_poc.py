import numpy as np
from pathlib import Path
import struct

# ---- dataset ----
def _read_idx_u8(path: Path) -> np.ndarray:
    """
    Reads MNIST-style IDX files (unsigned byte), returns a numpy array.
    Supports:
      - images: magic 2051, shape (N, rows, cols)
      - labels: magic 2049, shape (N,)
    """
    with path.open("rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            data = f.read(n * rows * cols)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)
            return arr
        elif magic == 2049:
            n = struct.unpack(">I", f.read(4))[0]
            data = f.read(n)
            arr = np.frombuffer(data, dtype=np.uint8)
            return arr
        else:
            raise ValueError(f"Unknown IDX magic {magic} in {path}")

def load_mnist_raw(root: str = "./data/mnist"):
    root = Path(root)

    train_images = _read_idx_u8(root / "train-images-idx3-ubyte")
    train_labels = _read_idx_u8(root / "train-labels-idx1-ubyte")
    test_images  = _read_idx_u8(root / "t10k-images-idx3-ubyte")
    test_labels  = _read_idx_u8(root / "t10k-labels-idx1-ubyte")

    # Convert to torch: flatten to (N,784), normalize to [0,1]
    X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
    y_train = train_labels.astype(np.int64)
    X_test  = test_images.reshape(test_images.shape[0], -1) / 255.0
    y_test  = test_labels.astype(np.int64)

    return (X_train, y_train,
            X_test,  y_test)

X_train, y_train, X_test, y_test = load_mnist_raw("./data/mnist")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def one_hot(y, num_classes=10):
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(logits):
    # stable softmax
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def cross_entropy_from_probs(p, y_onehot):
    # p: (m,10) probabilities
    eps = 1e-12
    return -np.sum(y_onehot * np.log(p + eps)) / p.shape[0]

# Parameters (He init for ReLU)
rng = np.random.default_rng(0)
W1 = (rng.standard_normal((784, 128)).astype(np.float32) * np.sqrt(2/784))
b1 = np.zeros((128,), dtype=np.float32)
W2 = (rng.standard_normal((128, 256)).astype(np.float32) * np.sqrt(2/128))
b2 = np.zeros((256,), dtype=np.float32)
W3 = (rng.standard_normal((256, 10)).astype(np.float32) * np.sqrt(2/256))
b3 = np.zeros((10,), dtype=np.float32)

lr = 3e-2
batch_size = 128

def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    logits = A2 @ W3 + b3
    P = softmax(logits)
    cache = (X, Z1, A1, Z2, A2, logits, P)
    return P, cache

def backward(y, cache, cur_lr):
    global W1, b1, W2, b2, W3, b3
    X, Z1, A1, Z2, A2, logits, P = cache
    m = X.shape[0]
    Y = one_hot(y, 10)  # (m,10)

    # dlogits = (P - Y)/m  (softmax + CE simplification)
    dZ3 = (P - Y) / m                     # (m,10)
    dW3 = A2.T @ dZ3                      # (256,10)
    db3 = dZ3.sum(axis=0)                 # (10,)

    dA2 = dZ3 @ W3.T                      # (m,256)
    dZ2 = dA2 * relu_grad(Z2)             # (m,256)
    dW2 = A1.T @ dZ2                      # (128,256)
    db2 = dZ2.sum(axis=0)                 # (256,)

    dA1 = dZ2 @ W2.T                      # (m,128)
    dZ1 = dA1 * relu_grad(Z1)             # (m,128)
    dW1 = X.T @ dZ1                       # (784,128)
    db1 = dZ1.sum(axis=0)                 # (128,)

    # SGD update
    W3 -= cur_lr * dW3; b3 -= cur_lr * db3
    W2 -= cur_lr * dW2; b2 -= cur_lr * db2
    W1 -= cur_lr * dW1; b1 -= cur_lr * db1

def accuracy(X, y, batch=512):
    correct = 0
    for i in range(0, X.shape[0], batch):
        P, _ = forward(X[i:i+batch])
        pred = P.argmax(axis=1)
        correct += (pred == y[i:i+batch]).sum()
    return correct / X.shape[0]

# Training loop
n = X_train.shape[0]
epochs = 20
for epoch in range(1, epochs + 1):

    # shuffle once per epoch
    perm = rng.permutation(n)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    cur_lr = lr * (1 - epoch/epochs * 0.5)

    epoch_loss = 0.0

    for i in range(0, n, batch_size):
        Xb = X_shuffled[i:i+batch_size]
        yb = y_shuffled[i:i+batch_size]

        P, cache = forward(Xb)
        loss = cross_entropy_from_probs(P, one_hot(yb, 10))
        backward(yb, cache, cur_lr)

        epoch_loss += loss * Xb.shape[0]

    epoch_loss /= n
    acc = accuracy(X_test, y_test)

    print(f"epoch {epoch:02d} | loss={epoch_loss:.4f} | test_acc={acc*100:.2f}%")