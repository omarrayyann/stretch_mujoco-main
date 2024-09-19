import matplotlib.pyplot as plt
import numpy as np

x = np.load("masks.npy")
print(x.shape)
plt.imshow(x)
plt.savefig("test.jpg")