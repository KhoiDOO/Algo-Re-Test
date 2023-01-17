import matplotlib.pyplot as plt
import numpy as np

my_dpi = 80

while True:
    f = plt.figure(figsize=(512/my_dpi, 512/my_dpi), dpi=my_dpi)
    plt.plot(np.random.randn(1000, 100))
    f.clear()
    plt.close(f)