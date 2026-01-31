import torch
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.pinnsformer import PINNsformer # pinnsformer_cfd should be root directory

class MyDataset(Dataset):
    def __init__(self, npz_file, patch_size):
        data = np.load(npz_file)
        self.X = data["X"].astype(np.float32)  # shape: [num_samples, seq_len, channels]
        self.Y = data["Y"].astype(np.float32)  # shape: [num_samples, seq_len, 1]
        self.patch_size = patch_size

        # Create patches per sample
        self.patches_X = []
        self.patches_Y = []
        for sample_idx in range(self.X.shape[0]):  # loop over samples
            x_sample = self.X[sample_idx]          # [seq_len, channels]
            y_sample = self.Y[sample_idx]          # [seq_len, 1]

            # Slice into patches along the sequence dimension
            for i in range(0, x_sample.shape[0]-patch_size+1, patch_size):
                x_patch = x_sample[i:i+patch_size]
                y_patch = y_sample[i:i+patch_size]

                self.patches_X.append(x_patch)
                self.patches_Y.append(y_patch)

    def __len__(self):
        return len(self.patches_X)

    def __getitem__(self, idx):
        # Returns: [patch_len, channels], [patch_len, 1]
        return self.patches_X[idx], self.patches_Y[idx]
    
def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load dataset ---
    dataset = MyDataset("pinnsformer_ready_data_part0.npz", patch_size=4092)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

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
            X_batch_attn = X_batch.permute(1, 0, 2)
            optimizer.zero_grad()
            Y_pred = model(X_batch_attn)
            Y_pred = Y_pred.permute(1, 0, 2)  
            loss = ((Y_pred - Y_batch)**2).mean()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pt")
        print(f"Epoch {epoch}, loss {loss.item()}")

if __name__ == "__main__":
    main()