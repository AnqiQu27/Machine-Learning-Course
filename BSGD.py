import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset.npy")
plt.figure()
for i, j in enumerate(data[2]):
    if j == 1:
        plt.scatter(data[0, i], data[1, i], c="r", marker="o")
    else:
        plt.scatter(data[0, i], data[1, i], c="b", marker="v")


train_x = np.array([data[0], data[1], np.ones(len(data[0]))])
train_y = data[2]
alpha = 0.1
lamb = 0.01
omega = np.array([[1, 1, 0]])
loss = np.array([])
for t in range(100):
    omega_k = omega[len(omega)-1]
    grad = 2 * lamb * omega_k
    loss_k = lamb * np.dot(omega_k, omega_k)
    for x,y in zip(train_x.T, train_y.T):
        grad = grad - y * np.exp(-y * np.dot(omega_k, x.T)) / (1 + np.exp(-y * np.dot(omega_k, x.T))) * x.T
        loss_k = loss_k + np.log(1 + np.exp(-y * np.dot(omega_k, x.T)))
    omega_k1 = omega_k - alpha * grad
    omega = np.append(omega, [omega_k1], axis=0)
    loss = np.append(loss, loss_k)

x1 = np.array([np.min(data[0]), np.max(data[0])])
x2 = - (omega[len(omega)-1, 0] * x1 + omega[len(omega)-1, 2])/omega[len(omega)-1, 1]
plt.plot(x1, x2)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14)
plt.show()

plt.figure()
plt.plot(np.arange(100), loss, label='Batch steepest gradient method')
plt.legend(['Newton method', 'Batch steepest gradient method'], loc = 'upper right')
plt.ylabel('$J(\omega^{(t)})$', fontsize = 14)
plt.xlabel('Number of iteration', fontsize = 14)
plt.show()



