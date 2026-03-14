import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from pathlib import Path
import numpy as np
import time

# ---- config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
lr = 0.1
epochs = 5

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

def load_mnist_raw(root, device):
    root = Path(root)

    train_images = _read_idx_u8(root / "train-images-idx3-ubyte")
    train_labels = _read_idx_u8(root / "train-labels-idx1-ubyte")
    test_images  = _read_idx_u8(root / "t10k-images-idx3-ubyte")
    test_labels  = _read_idx_u8(root / "t10k-labels-idx1-ubyte")

    # Convert to torch: flatten to (N,784), normalize to [0,1]
    X_train = torch.from_numpy(train_images).to(torch.float32).view(-1, 784) / 255.0
    y_train = torch.from_numpy(train_labels.astype(np.int64))
    X_test  = torch.from_numpy(test_images).to(torch.float32).view(-1, 784) / 255.0
    y_test  = torch.from_numpy(test_labels.astype(np.int64))

    device = torch.device(device)
    return (X_train.to(device), y_train.to(device),
            X_test.to(device),  y_test.to(device))

X_train, y_train, X_test, y_test = load_mnist_raw("./data/mnist", device)

X_val = X_train[50000:]
y_val = y_train[50000:]
X_train = X_train[:50000]
y_train = y_train[:50000]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# ---- model ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(784, 128))
        self.b1 = nn.Parameter(torch.zeros(128))
        self.W2 = nn.Parameter(torch.empty(128, 256))
        self.b2 = nn.Parameter(torch.zeros(256))
        self.W3 = nn.Parameter(torch.empty(256, 10))
        self.b3 = nn.Parameter(torch.zeros(10))
        # He init for ReLU
        nn.init.kaiming_normal_(self.W1, nonlinearity="relu")
        nn.init.kaiming_normal_(self.W2, nonlinearity="relu")
        nn.init.kaiming_normal_(self.W3, nonlinearity="linear")

    def forward(self, x):
        # x: (m,784)
        a1 = F.relu(x @ self.W1 + self.b1)      # (m,128)
        a2 = F.relu(a1 @ self.W2 + self.b2)     # (m,256)
        logits = a2 @ self.W3 + self.b3         # (m,10) raw logits
        return logits

model = MLP().to(device)

# CrossEntropyLoss expects raw logits (no softmax here)
criterion = nn.CrossEntropyLoss(reduction="mean")

@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(X_val)
    loss = criterion(logits, y_val).item()
    acc = (logits.argmax(dim=1) == y_val).float().mean().item()
    return loss, acc

def train_one_epoch():
    model.train()
    n = X_train.size(0)
    perm = torch.randperm(n, device=device)

    total_loss = 0.0
    seen = 0

    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()

        # ---- manual parameter update (no optimizer) ----
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                p.grad = None  # clear grads

        bs = xb.size(0)
        total_loss += loss.item() * bs
        seen += bs

    return total_loss / seen

start_time = time.time()
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch()
    test_loss, test_acc = evaluate()
    print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={test_loss:.4f} | val_acc={test_acc*100:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")