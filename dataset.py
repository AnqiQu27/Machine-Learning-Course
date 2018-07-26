import numpy as np
import matplotlib.pyplot as plt

omega = np.random.randn()


def generate_data(n):
    noise = 0.8 * np.random.randn(1,n)
    x = np.random.randn(2,n)
    y = 2 * (omega * x[0] + x[1] + noise > 0) - 1
    return np.r_[x, y]


a = generate_data(500)
np.save("dataset",a)