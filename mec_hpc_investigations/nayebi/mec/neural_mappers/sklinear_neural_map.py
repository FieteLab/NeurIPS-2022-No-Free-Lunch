import numpy as np
import sklearn.linear_model as lm

from . import neural_map_base as nm

__all__ = ["SKLinearNeuralMap"]

class SKLinearNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing linear regression between source and target.
    """
    def __init__(self, regression_type="Ridge", regression_kwargs={},
                 weights_save_nm=None):
        super(SKLinearNeuralMap, self).__init__()
        self._regression_type = regression_type
        self._regression_kwargs = regression_kwargs
        self._mapper = None
        self._weights_save_nm = weights_save_nm
        self._weights = None

    def fit(self, X, Y):
        # If target is a one-dimensional vector, make it into a column vector
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        assert X.shape[0] == Y.shape[0], \
            f"Features and targets sample dimension does not match: {X.shape} and {Y.shape}"
        assert X.ndim == Y.ndim == 2

        self._n_source = X.shape[1]
        self._n_targets = Y.shape[1]

        self._mapper = getattr(lm, self._regression_type)(**self._regression_kwargs)
        self._mapper.fit(X, Y)
        if self._weights_save_nm is not None:
            self._weights = self._mapper.coef_
            # save after fitting
            print(f"Saving weights to {self._weights_save_nm}")
            np.savez(self._weights_save_nm, self._weights)

        self._fitted = True

    def predict(self, X):
        """
        Performs response prediction with features from test set.

        Inputs:
            X : (numpy.ndarray) of dimensions (num_stimuli, num_features)

        Returns:
            preds : (numpy.ndarray) of dimensions (num_stimuli, num_targets)
        """
        assert self._n_source is not None, "Fit procedure was not run."
        assert self._n_targets is not None, "Fit procedure was not run."
        assert self._mapper is not None, "Fit procedure was not run."
        assert self._fitted == True, "Fitting failed."
        assert X.shape[1] == self._n_source, \
            f"Number of train features ({self._n_source}) does not align with number of test features ({X.shape[1]})"

        preds = self._mapper.predict(X)
        if preds.ndim == 1:
            preds = preds[:,np.newaxis]
        return preds


