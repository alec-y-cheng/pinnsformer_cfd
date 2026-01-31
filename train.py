import torch
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pinnsformer_cfd.model.pinnsformer import PINNsformer # or whatever the repo uses

class MyDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load dataset ---
    dataset = MyDataset("pinnsformer_ready_data_part0.npz")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

    # --- Initialize model ---
    model = PINNsformer(d_out = dataset.Y.shape[-1],
                        d_model = 256,
                        d_hidden = 1024,
                        N = 6, 
                        heads = 8)

    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Training loop ---
    for epoch in range(10):
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = ((Y_pred - Y_batch)**2).mean()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pt")
        print(f"Epoch {epoch}, loss {loss.item()}")

if __name__ == "__main__":
    main()