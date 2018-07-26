import numpy as np
import numpy.linalg as la
import  matplotlib.pyplot as plt

A = np.mat([[3.0, 0.5],[0.5, 1.0]])
myu = np.mat([[1.0], [2.0]])
L = max(la.eigvals(2 * A))
lamb = 4
q = lamb / L


def grad(omega_k):
    return np.array((2 * A * (np.mat(omega_k).T - myu)).T)[0]


omega = np.array([[1.0, 1.0]])
s = np.array([1.0])
v = np.array([1.0, 1.0])
for t in range(1000):
    omega_k = omega[len(omega) - 1]
    omega_k1 = np.array([0.0, 0.0])
    z = v - 1 / L * grad(v)
    for i in range(len(omega_k)):
        if z[i] > q:
            omega_k1[i] = z[i] - q
        elif z[i] < -q:
            omega_k1[i] = z[i] + q
        else:
            omega_k1[i] = 0
    omega = np.append(omega, [omega_k1], axis=0)
    sk = s[len(s)-1]
    sk1 = 1 / 2 + np.sqrt(1 + 4 * np.square(sk)) / 2
    s = np.append(s,sk1)
    qk = (sk - 1) / sk1
    v = omega_k1 + qk * (omega_k1 - omega_k)

omega_hat = omega[len(omega)-1]
print(omega_hat)
norm = la.norm(omega - omega_hat, 2, axis=-1)
plt.figure()
plt.semilogy(range(800), norm[range(800)], label= "Accelerated proximal gradient")
# plt.legend(['Proximal gradient', 'Accelerated proximal gradient'], loc = 'upper right')
plt.ylabel(r'$\||\omega^{t}-\hat{\omega}\||$', fontsize = 14)
plt.xlabel('Number of iteration', fontsize = 14)
plt.show()



