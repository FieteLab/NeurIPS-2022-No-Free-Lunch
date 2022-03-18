import numpy as np
from collections import OrderedDict
import os, copy
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from sklearn.metrics import r2_score
from tqdm import tqdm
from mec_hpc_investigations.core.constants import CAITLIN1D_VR_SESSIONS
from mec_hpc_investigations.core.default_dirs import BASE_DIR_RESULTS
from mec_hpc_investigations.neural_data.utils import trialbatch_1dvr_file_loader, animal_session_1dvr

def population_normalize_1dvr(frs_trials,
                              clip=90, normalize=True,
                              flatten=False):
    assert(len(frs_trials.shape) == 3)
    # procedure from Low et al. 2020, pg. 24
    if clip is not None:
        # clip per neuron
        max_clip_val = np.percentile(frs_trials, q=clip, axis=(0, 1), keepdims=True)
        min_clip_val = np.zeros_like(max_clip_val)
        frs_trials = np.clip(frs_trials, a_min=min_clip_val, a_max=max_clip_val)
        # sanity check that it clipped neurons
        assert(np.array_equal(np.squeeze(max_clip_val), np.amax(frs_trials, axis=(0, 1))) is True)

    if normalize:
        min_per_neuron = np.amin(frs_trials, axis=(0, 1), keepdims=True)
        max_per_neuron = np.amax(frs_trials, axis=(0, 1), keepdims=True)
        frs_trials = (frs_trials - min_per_neuron)/(max_per_neuron - min_per_neuron)
        frs_trials = np.nan_to_num(frs_trials)

    if flatten:
        frs_trials = frs_trials.reshape((frs_trials.shape[0], -1))

    return frs_trials

def trial_by_trial_similarity(frs_trials):
    frs_trials_vec = frs_trials.reshape((frs_trials.shape[0], -1))
    return np.corrcoef(frs_trials_vec)

# the code below up to the cv_kmeans() function is adapted from: https://gist.github.com/ahwillia/65d8f87fcd4bded3676d67b55c1a3954
# see this blog post: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/ for details

def censored_nnlstsq(A, B, M):
    """Solves nonnegative least-squares problem with missing data in B
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : nonnegative r x n matrix that minimizes norm(M*(AX - B))
    """
    if A.ndim == 1:
        A = A[:,None]
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    X = np.empty((B.shape[1], A.shape[1]))
    for n in range(B.shape[1]):
        X[n] = nnlstsq(T[n], rhs[n], is_input_prod=True)[0].T
    return X.T

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
    except:
        r = T.shape[1]
        T[:,np.arange(r),np.arange(r)] += 1e-6
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T

def cv_pca(data, rank, M=None, M_seed=0,
           p_holdout=0.1, nonneg=False, max_iter=300):
    """Fit PCA or NMF while holding out a fraction of the dataset.
    """

    # choose solver for alternating minimization
    if nonneg:
        solver = censored_nnlstsq
    else:
        solver = censored_lstsq

    # create masking matrix
    if M is None:
        M = np.random.RandomState(M_seed).rand(*data.shape) > p_holdout

    # initialize U randomly
    if nonneg:
        U = np.random.rand(data.shape[0], rank)
    else:
        U = np.random.randn(data.shape[0], rank)

    # fit pca/nmf
    for itr in tqdm(range(max_iter)):
        Vt = solver(U, data, M)
        U = solver(Vt.T, data.T, M.T).T

    # return result and test/train error
    pred = np.dot(U, Vt)
    resid = pred - data
    metrics = {}
    metrics["train_err"] = np.mean(resid[M]**2)
    try:
        metrics["test_err"] = np.mean(resid[~M]**2)
    except:
        pass
    metrics["train_rsq"] = r2_score(y_true=data[M], y_pred=pred[M])
    try:
        metrics["test_rsq"] = r2_score(y_true=data[~M], y_pred=pred[~M])
    except:
        pass
    return U, Vt, metrics

def cv_kmeans(data, rank, p_holdout=.1,
              M=None, M_seed=0,
              max_iter=300, tol=None):
    """Fit kmeans while holding out a fraction of the dataset.
    """

    # create masking matrix
    if M is None:
        M = np.random.RandomState(M_seed).rand(*data.shape) > p_holdout

    # initialize cluster centers
    Vt = np.random.randn(rank, data.shape[1])
    U = np.empty((data.shape[0], rank))
    rn = np.arange(U.shape[0])

    # initialize missing data randomly
    imp = data.copy()
    imp[~M] = np.random.randn(*data.shape)[~M]

    # initialize cluster centers far apart
    Vt = [imp[np.random.randint(data.shape[0])]]
    while len(Vt) < rank:
        i = np.argmax(np.min(cdist(imp, Vt), axis=1))
        Vt.append(imp[i])
    Vt = np.array(Vt)

    # fit kmeans
    prev_train_err = 0
    for itr in tqdm(range(max_iter)):
        # update cluster assignments
        clus = np.argmin(cdist(imp, Vt), axis=1)
        U.fill(0.0)
        U[rn, clus] = 1.0
        # update centroids
        Vt = censored_lstsq(U, imp, M)
        assert np.all(np.sum(np.abs(Vt), axis=1) > 0)
        # update estimates of missing data
        imp[~M] = np.dot(U, Vt)[~M]
        # compute residual
        resid = np.dot(U, Vt) - data
        curr_train_err = np.mean(resid[M]**2)
        if tol is not None:
            if np.abs(curr_train_err - prev_train_err) < tol:
                break
            else:
                prev_train_err = curr_train_err

    # return result and test/train error
    pred = np.dot(U, Vt)
    resid = pred - data
    metrics = {}
    metrics["train_err"] = np.mean(resid[M]**2)
    try:
        metrics["test_err"] = np.mean(resid[~M]**2)
    except:
        pass
    metrics["train_rsq"] = r2_score(y_true=data[M], y_pred=pred[M])
    try:
        metrics["test_rsq"] = r2_score(y_true=data[~M], y_pred=pred[~M])
    except:
        pass
    return clus, U, Vt, metrics

# end of unsupervised methods code

def compute_num_maps(sess_data,
                     clip=None, normalize=False,
                     shuffle_seed=0,
                     p_holdout=0.1,
                     num_holdouts=10,
                     ranks=None,
                     max_rank=None,
                     max_iter=100,
                     verbose=False,
                     avg_shuffle_rsq_thresh=0.7,
                     avg_test_rsq_thresh=0.63):

    break_on_criterion = False
    if max_rank is None:
        break_on_criterion = True
        # min over trials, position bins, and neurons
        max_rank = np.amin(population_normalize_1dvr(sess_data,
                                                clip=clip,
                                                normalize=normalize).shape) - 1

    X = population_normalize_1dvr(sess_data,
                                  clip=clip,
                                  normalize=normalize,
                                  flatten=True)
    Q = ortho_group.rvs(dim=X.shape[0],
                        random_state=np.random.RandomState(shuffle_seed))
    X_shuffle = np.matmul(Q, X)
    metrics = {"ranks": [], "kmeans_test_rsq": [],
               "shufflekmeans_test_rsq": [],
               "pca_test_rsq": [],
               "mean_shuffle_ratio": []}

    if ranks is None:
        ranks = range(1, max_rank+1)
    for r in ranks:
        if verbose:
            print("Curr rank", r)
        metrics["ranks"].append(r)
        k_means_test_rsq = []
        shuffle_k_means_test_rsq = []
        pca_test_rsq = []
        for s in range(num_holdouts):
            _, _, _, k_means_metric = cv_kmeans(X,
                                             rank=r,
                                             p_holdout=p_holdout,
                                             M_seed=s,
                                             max_iter=max_iter)
            k_means_test_rsq.append(k_means_metric["test_rsq"])

            _, _, _, shuffle_k_means_metric = cv_kmeans(X_shuffle,
                                             rank=r,
                                             p_holdout=p_holdout,
                                             M_seed=s,
                                             max_iter=max_iter)
            shuffle_k_means_test_rsq.append(shuffle_k_means_metric["test_rsq"])

            _, _, pca_metric = cv_pca(X,
                                         rank=r,
                                         p_holdout=p_holdout,
                                         M_seed=s,
                                         max_iter=max_iter)
            pca_test_rsq.append(pca_metric["test_rsq"])

        k_means_test_rsq = np.array(k_means_test_rsq)
        metrics["kmeans_test_rsq"].append(k_means_test_rsq)
        shuffle_k_means_test_rsq = np.array(shuffle_k_means_test_rsq)
        metrics["shufflekmeans_test_rsq"].append(shuffle_k_means_test_rsq)
        pca_test_rsq = np.array(pca_test_rsq)
        metrics["pca_test_rsq"].append(pca_test_rsq)

        shuffle_ratio = (k_means_test_rsq - shuffle_k_means_test_rsq) / (pca_test_rsq - shuffle_k_means_test_rsq)
        metrics["mean_shuffle_ratio"] = np.mean(shuffle_ratio)

        if verbose:
            print(f"Rank {r}, KMeans Avg Test RSq {np.mean(k_means_test_rsq)}, Mean Shuffle Ratio {np.mean(shuffle_ratio)}")

        criterion = (np.mean(k_means_test_rsq) >= avg_test_rsq_thresh) and (np.mean(shuffle_ratio) < avg_shuffle_rsq_thresh)
        if criterion and break_on_criterion:
            break

    return metrics

def get_1dvr_session_trial_assignments(sess_data,
                                     sess_clus="precomputed",
                                     sess_num_maps=None,
                                     clip=None,
                                     normalize=False,
                                     max_iter=100):

    """Computes map assignments based on k means clustering.
    Where the number of (> 1) centroids/maps is set by sess_num_maps.
    If you want to not do any clustering and use all of the trials,
    either set sess_clus={} or sess_num_maps={}. The former is recommended
    since that is more direct."""

    sessions = list(sess_data.keys())

    if sess_num_maps is None:
        sess_num_maps = {4: 4,
                         5: 3,
                         6: 2,
                         7: 2,
                         8: 2,
                         9: 2}
    else:
        # if specifying your own num maps, you shouldn't call precomputed
        assert(sess_clus != "precomputed")

    if sess_clus is None:
        sess_clus = {}
        for sess_name in sess_num_maps.keys():
            if sess_name in sessions:
                print(f"Computing {sess_num_maps[sess_name]} cluster assignments for session {sess_name}")
                X = population_normalize_1dvr(sess_data[sess_name],
                                              clip=clip,
                                              normalize=normalize,
                                              flatten=True)
                # fit to all of the data
                M = np.ones_like(X).astype(bool)
                clus, _, _, _ = cv_kmeans(X,
                                          rank=sess_num_maps[sess_name],
                                          M=M,
                                          max_iter=max_iter)
                sess_clus[sess_name] = clus

        # deal with spurious clusters in session 4 if you want multiple maps of it
        if (4 in sessions) and (4 in sess_num_maps):
            # we nest this if statement otherwise if the first statement is not true might give index error
            if (len(np.unique(sess_clus[4])) == 4):
                print("Dealing with spurious clusters in Session 4")
                two_loc = np.where(sess_clus[4] == 2)[0]
                one_loc = np.where(sess_clus[4] == 1)[0]
                sess_clus[4][two_loc] = 0
                sess_clus[4][one_loc] = 0
                three_loc = np.where(sess_clus[4] == 3)[0]
                # change back to 1, since now we have two maps
                assert(len(np.unique(sess_clus[4])) == 2)
                sess_clus[4][three_loc] = 1

    elif sess_clus == "precomputed":
        print("Loading precomputed cluster assignments")
        # since we computed the assignments with these settings
        assert((clip is None) and (normalize is False))
        sess_clus = np.load(os.path.join(BASE_DIR_RESULTS, "caitlin1d_vr_session_trial_cluster_assignments.npz"),
                            allow_pickle=True)['arr_0'][()]

    # add single map sessions
    for s in sessions:
        if s not in sess_clus.keys():
            # sanity check that this is indeed a single map session, rather than a missed session
            assert(s not in sess_num_maps.keys())
            sess_clus[s] = np.zeros(sess_data[s].shape[0]).astype(np.int64)

    return sess_clus

def aggregate_responses_1dvr(sessions=CAITLIN1D_VR_SESSIONS,
                             animal_session_precomputed=True,
                             sess_clus="precomputed",
                             sess_num_maps=None,
                             clip=None,
                             normalize=False,
                             kmeans_max_iter=100):

    sess_data = OrderedDict()
    for sess_name in sessions:
        sess_data[sess_name] = trialbatch_1dvr_file_loader(session_name=sess_name)

    sess_clus = get_1dvr_session_trial_assignments(sess_data=sess_data,
                                                     sess_clus=sess_clus,
                                                     sess_num_maps=sess_num_maps,
                                                     clip=clip,
                                                     normalize=normalize,
                                                     max_iter=kmeans_max_iter)

    animal_sess_mapping = animal_session_1dvr(sessions=sessions,
                                              precomputed=animal_session_precomputed)

    spec_resp_agg  = OrderedDict()
    for spec, spec_sessions in animal_sess_mapping.items():
        spec_resp_agg[spec] = OrderedDict()
        for s in spec_sessions:
            spec_resp_agg[spec][s] = []
            curr_sess_maps = list(np.sort(np.unique(sess_clus[s])))
            total_trials = sess_data[s].shape[0]
            curr_total_trials = 0
            for m in curr_sess_maps:
                curr_map_idx = np.where(sess_clus[s] == m)[0]
                curr_map_resp = sess_data[s][curr_map_idx]
                if len(curr_map_idx) == 1:
                    assert(len(curr_map_resp.shape) == 2)
                    curr_map_resp = np.expand_dims(curr_map_resp, axis=0)
                curr_total_trials += curr_map_resp.shape[0]
                spec_resp_agg[spec][s].append(curr_map_resp)
            # sanity check that we have accounted for all of the trials
            assert(curr_total_trials == total_trials)
    return spec_resp_agg

def pad_1dvr_trials(population_maps):
    """Input `population_maps` is a list of numpy arrays of trials x position bins x neurons maps.
    These maps can come from different animals or the same animal but different recording sessions.
    What is returned is the concatenation across neurons of the maps in this list, after the trials
    are padded up to the maximum number of trials present in the population_maps list."""

    assert(isinstance(population_maps, list))
    max_trials = np.amax([map_resp.shape[0] for map_resp in population_maps])
    population_maps_concat = []
    for map_resp in population_maps:
        pad_trials = max_trials - map_resp.shape[0]
        if pad_trials > 0:
            pad_arr = np.zeros([pad_trials] + list(map_resp.shape[1:])) + np.nan
            population_maps_concat.append(np.concatenate([map_resp, pad_arr], axis=0))
        else:
            population_maps_concat.append(map_resp)
    # concatenate across neurons
    population_maps_concat = np.concatenate(population_maps_concat, axis=-1)
    return population_maps_concat
