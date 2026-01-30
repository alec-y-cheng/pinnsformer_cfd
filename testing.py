import numpy as np

npz = np.load('full_dataset_2000.npz', mmap_mode='r')

print(npz.files)

x = npz['X']
y = npz['Y']
c = npz['channel_names']

#print(x.shape)
print(y.shape)
print(c.shape)
print(c)

print(y[0, 0, 100, :])