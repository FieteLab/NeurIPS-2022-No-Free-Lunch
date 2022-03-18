import numpy as np

from scipy.stats import pearsonr

from mec_hpc_investigations.neural_data.metrics import rsa

__all__ = ["NeuralMapBase"]

class NeuralMapBase():
    def __init__(self, score_func="corr"):
        self._score_func = score_func
        self._fitted = False
        self._n_source = None
        self._n_targets = None

    def score(self, Y, Y_pred,
              score_func=None):

        if score_func is None:
            score_func = self._score_func

        if score_func == "corr":
            scores = self._correlation_score(Y, Y_pred)
        elif score_func == "rsa":
            scores = self._rsa_score(Y, Y_pred)
        else:
            raise ValueError(f"{self._score_func} is not a supported score function.")

        return scores

    def _correlation_score(self, Y, Y_pred):
        """
        Score the predictions with the correlation metric.

        Input:
            Y      : (numpy.ndarray) (n_samples, n_targets) for actual values
                     (i.e., targets).
            Y_pred : (numpy.ndarray) (n_samples, n_targets) for predictions.

        Output:
            score  : (float) Pearson correlation across targets
        """
        assert Y.ndim == Y_pred.ndim == 2
        assert Y.shape == Y_pred.shape

        corrs = np.zeros((Y.shape[1],)) + np.NaN
        for i in range(Y.shape[1]):
            if np.isnan(Y_pred[:,i]).sum() == 0:
                _corr = pearsonr(Y[:,i], Y_pred[:,i])[0]
            else:
                _corr = np.NaN
            corrs[i] = _corr

        return corrs

    def _negemd_score(self, Y, Y_pred, M):
        """
        Score the predictions with the negative EMD metric.

        Input:
            Y      : (numpy.ndarray) (n_samples, n_targets) for actual values
                     (i.e., targets).
            Y_pred : (numpy.ndarray) (n_samples, n_targets) for predictions.
            M : (np.ndarray) (n_samples, n_samples) distance matrix.

        Output:
            score  : (float) Negative EMD across targets
        """
        from . utils import negemd
        assert Y.ndim == Y_pred.ndim == 2
        assert Y.shape == Y_pred.shape

        corrs = np.zeros((Y.shape[1],)) + np.NaN
        for i in range(Y.shape[1]):
            if np.isnan(Y_pred[:,i]).sum() == 0:
                _corr = negemd(x=Y[:, i], y=Y_pred[:, i], M=M)
            else:
                _corr = np.NaN
            corrs[i] = _corr

        return corrs

    def _rsa_score(self, Y, Y_pred):
        """
        Score the predictions with the RSA metric.

        Input:
            Y      : (numpy.ndarray) (n_samples, n_sources) for actual values
                     (i.e., targets).
            Y_pred : (numpy.ndarray) (n_samples, n_targets) for predictions.

        Output:
            score  : (float) RDM correlation
        """
        assert Y.ndim == Y_pred.ndim == 2
        assert Y.shape[0] == Y_pred.shape[0]

        # If any predictions are NaN, return NaN score
        if np.isnan(Y_pred).sum() != 0:
            return np.NaN

        score = rsa(Y, Y_pred)
        return score

    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


