import numpy as np


sigma_e = np.arange(5, 101) / 100.
surround_scale = np.arange(15, 101, 5) / 10.


def calc_k_star(sigma, s):
    k_star_squared = 6. * np.log(s) / (s * s - 1.) / np.square(sigma)
    k_star = np.sqrt(k_star_squared)
    return k_star


sigmasigma, ss = np.meshgrid(sigma_e, surround_scale)
k_star = calc_k_star(sigmasigma, ss)
k_star = np.round(k_star)


import matplotlib.pyplot as plt
h = plt.contourf(sigma_e, surround_scale, k_star, levels=np.arange(0, 30),
                 cmap="rainbow")
plt.colorbar()
plt.xlabel(r'$\sigma_E$ (m)')
plt.ylabel(r'$s = \sigma_I / \sigma_E $')
plt.title(r'$k^*$')
plt.show()

