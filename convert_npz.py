import numpy as np

def convert_npz(batch_size, npz_file="full_dataset_2000.npz"):
    # - channels = ['SDF' 'Bldg_height' 'Z_relative' 'U_over_Uref' 'X_local' 'Y_local' 'dir_sin' 'dir_cos']
    data = np.load(npz_file)
    X = data["X"]  # [batch, channels=8, H, W]
    Y = data["Y"]  # [batch, 1, H, W]

    # --- Check batch dimensions ---
    if Y.ndim == 3:  # expand batch dimension if needed
        Y = np.expand_dims(Y, axis=0)

    print("Original X shape:", X.shape)
    print("Original Y shape:", Y.shape)
    
    # Remove the singleton dimension
    X = np.squeeze(X, axis=1)  # now [2000, 8, 504, 504]
    Y = np.squeeze(Y, axis=1)  # now [2000, 504, 504]

    batch_size, channels, H, W = X.shape
    seq_len = H * W

    # --- Rearrange channels: coordinates first ---
    # Assuming original X channels order: [SDF, Bldg_height, Z_relative, U_at_z, X_coords, Y_coords, dir_sin, dir_cos]
    # We want: [X_coords, Y_coords, SDF, Bldg_height, Z_relative, U_at_z, dir_sin, dir_cos]
    X_ordered = np.stack([
        X[:, 4, :, :],  # X_coords
        X[:, 5, :, :],  # Y_coords
        X[:, 0, :, :],  # SDF
        X[:, 1, :, :],  # Bldg_height
        X[:, 2, :, :],  # Z_relativechle kim
        X[:, 3, :, :],  # U_at_z
        X[:, 6, :, :],  # dir_sin
        X[:, 7, :, :]   # dir_cos
    ], axis=1)  # shape [batch, 8, H, W]

    # --- Flatten spatial dimensions and move channels last ---
    X_seq = np.transpose(X_ordered, (0, 2, 3, 1))  # [batch, H, W, channels]
    X_seq = X_seq.reshape(batch_size, seq_len, channels)
    print("Converted X shape:", X_seq.shape)

    # Y: flatten similarly
    Y_seq = np.transpose(Y[..., np.newaxis], (0, 2, 3, 1))  # [batch, H, W, 1]
    Y_seq = Y_seq.reshape(batch_size, seq_len, 1)
    print("Converted Y shape:", Y_seq.shape)

        # Save in batches
    for i in range(0, X_seq.shape[0], batch_size):
        start = i
        end = min(i + batch_size, X_seq.shape[0])
        X_batch = X_seq[start:end]
        Y_batch = Y_seq[start:end]
        filename = f"pinnsformer_ready_data_part{start}.npz"
        np.savez_compressed(filename, X=X_batch, Y=Y_batch)
        print(f"Saved {filename} with samples {start}-{end-1}")

    print("All batches saved!")

if __name__ == "__main__":
    convert_npz(batch_size=500)
    print("done")