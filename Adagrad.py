import numpy as np
import numpy.linalg as la
import  matplotlib.pyplot as plt

A = np.mat([[250.0, 15.0],[15.0, 4.0]])
myu = np.mat([[1.0], [2.0]])
L = max(la.eigvals(2 * A))
lamb = 0.89
eta = 500 / L
delta = 0.02


def grad(omega_k):
    return np.array((2 * A * (np.mat(omega_k).T - myu)).T)[0]


omega = np.array([[1.0, 1.0]])
g = np.array([0.0, 0.0])
G = np.array([0.0, 0.0])
for t in range(1000):
    omega_k = omega[len(omega) - 1]
    omega_k1 = np.array([0.0, 0.0])
    z = np.array([0.0, 0.0])
    for i in range(len(omega_k)):
        g[i] = grad(omega_k)[i]
        G[i] = G[i] + grad(omega_k)[i] * grad(omega_k)[i]
        z[i] = omega_k[i] - eta * g[i] / (np.sqrt(G[i]) + delta)
        q = lamb * eta / (np.sqrt(G[i]) + delta)
        if z[i] > q:
            omega_k1[i] = z[i] - q
        elif z[i] < -q:
            omega_k1[i] = z[i] + q
        else:
            omega_k1[i] = 0
    omega = np.append(omega, [omega_k1], axis=0)

omega_hat = omega[len(omega)-1]
print(omega_hat)
norm = la.norm(omega - omega_hat, 2, axis=-1)
plt.semilogy(range(500), norm[range(500)], label= "Adagrad")
# plt.legend(['Proximal gradient', 'Accelerated proximal gradient', 'Adagrad'], loc = 'upper right')
plt.ylabel(r'$\||\omega^{t}-\hat{\omega}\||$', fontsize = 14)
plt.xlabel('Number of iteration', fontsize = 14)
plt.show()



