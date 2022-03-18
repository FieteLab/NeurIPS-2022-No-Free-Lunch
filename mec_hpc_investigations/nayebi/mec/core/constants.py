import numpy as np

CAITLIN1D_VR_SESSIONS = range(1, 12)

gridscore_starts = [0.2] * 10
gridscore_ends = np.linspace(0.4, 1.0, num=10)

SVM_CV_C_LONG = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]
RIDGE_CV_ALPHA_LONG = list(1.0/np.array(SVM_CV_C_LONG)) + [1]

ALPHA_RNG = list(np.geomspace(1e-9, 1e9, num=99, endpoint=True)) # note: this includes 1.0
L1_RATIO_RNG = ([1e-16] + list(np.linspace(0.01, 1.0, num=99, endpoint=True)))

ALPHA_RNG_1D = list(np.geomspace(1e-9, 1e9, num=5, endpoint=True)) # note: this includes 1.0
L1_RATIO_RNG_1D = ([1e-16] + list(np.linspace(0.01, 1.0, num=4, endpoint=True)))

ALPHA_RNG_SHORT = list(np.geomspace(1e-9, 1e9, num=9, endpoint=True)) # note: this includes 1.0
L1_RATIO_RNG_SHORT = ([1e-16] + list(np.linspace(0.01, 1.0, num=9, endpoint=True)))