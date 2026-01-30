import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pinnsformer_cfd import PINNsFormerModel  # or whatever the repo uses

class MyDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- Load dataset ---
dataset = MyDataset("pinnsformer_ready_data.npz")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Initialize model ---
model = PINNsFormerModel(...)  # fill in repo-specific args

# --- Optimizer ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
for epoch in range(10):
    for X_batch, Y_batch in loader:
        X_batch = X_batch.cuda()
        Y_batch = Y_batch.cuda()
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = ((Y_pred - Y_batch)**2).mean()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, loss {loss.item()}")
