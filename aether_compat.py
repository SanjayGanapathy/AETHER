"""
aether_compat.py
----------------
Optional-dependency fallbacks for the AETHER framework.

AETHER.py prefers `hmmlearn` and `shap` when they are installed. This module
supplies drop-in replacements so that the pipeline is runnable, and therefore
reproducible, in environments where those two packages are unavailable.

Two components:

1. GaussianHMMFallback
   A diagonal-covariance Gaussian hidden Markov model exposing the subset of
   the hmmlearn API that AETHER uses: fit(X, lengths), startprob_, transmat_,
   means_, covars_, n_components. Trained by Baum-Welch in log space. The
   forward, backward and expected-transition steps are vectorised across
   sequences, so the Python-level loop runs once per time step of the longest
   sequence rather than once per observation. Multi-individual data passed via
   `lengths` is handled correctly and no transition is ever accumulated across
   a sequence boundary.

2. exact_shapley
   Exact Shapley values by full enumeration of feature coalitions. Each AETHER
   expert uses only three or four features, so the 2**|F| coalitions can be
   enumerated directly. This is exact where KernelSHAP is a sampling
   approximation, and it removes a dependency.

Author: Sanjay Ganapathy
"""

import itertools
import math

import numpy as np
from scipy.special import logsumexp

__all__ = ["GaussianHMMFallback", "exact_shapley"]

_LOG_FLOOR = 1e-300


# ----------------------------------------------------------------------------
# 1. Gaussian HMM (diagonal covariance)
# ----------------------------------------------------------------------------

class GaussianHMMFallback:
    """Diagonal-covariance Gaussian HMM trained by Baum-Welch in log space."""

    def __init__(self, n_components=4, covariance_type="diag", n_iter=100,
                 tol=1e-4, random_state=55, min_covar=1e-3, verbose=False,
                 max_batch_cells=8_000_000):
        if covariance_type != "diag":
            raise ValueError("GaussianHMMFallback supports covariance_type='diag' only.")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.min_covar = float(min_covar)
        self.verbose = verbose
        self.max_batch_cells = int(max_batch_cells)

    # -- sequence bookkeeping ------------------------------------------------

    @staticmethod
    def _bounds(X, lengths):
        n = len(X)
        if lengths is None:
            return np.array([0, n])
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.sum() != n:
            raise ValueError(f"lengths sum to {lengths.sum()} but X has {n} rows.")
        return np.concatenate([[0], np.cumsum(lengths)])

    def _batches(self, bounds):
        """Group sequences of similar length into padded batches.

        Returns a list of (starts, ends) arrays. Sorting by length first keeps
        the padding waste small.
        """
        starts, ends = bounds[:-1], bounds[1:]
        lens = ends - starts
        order = np.argsort(-lens)
        k = self.n_components
        out, cur = [], []
        cur_max = 0
        for i in order:
            new_max = max(cur_max, int(lens[i]))
            if cur and (len(cur) + 1) * new_max * k > self.max_batch_cells:
                out.append(np.array(cur))
                cur, cur_max = [i], int(lens[i])
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            out.append(np.array(cur))
        return [(starts[b], ends[b]) for b in out]

    @staticmethod
    def _pad(X, starts, ends):
        """Stack sequences into (B, Tmax, D) with a boolean validity mask."""
        lens = ends - starts
        b, t_max, d = len(starts), int(lens.max()), X.shape[1]
        out = np.zeros((b, t_max, d), dtype=float)
        mask = np.zeros((b, t_max), dtype=bool)
        for i, (s, e) in enumerate(zip(starts, ends)):
            out[i, : e - s] = X[s:e]
            mask[i, : e - s] = True
        return out, mask

    # -- emissions -----------------------------------------------------------

    def _log_emission(self, Xb):
        """log N(x | mean_k, diag(var_k)) for a padded batch, shape (B, T, K).

        Expanded into matrix products so no (B, T, K, D) tensor is formed.
        """
        var, mean = self.covars_, self.means_
        inv = 1.0 / var                                  # (K, D)
        const = -0.5 * (Xb.shape[-1] * np.log(2.0 * np.pi) + np.log(var).sum(axis=1))
        quad = (
            (Xb ** 2) @ inv.T
            - 2.0 * (Xb @ (mean * inv).T)
            + ((mean ** 2) * inv).sum(axis=1)[None, None, :]
        )
        return const[None, None, :] - 0.5 * quad

    # -- initialisation ------------------------------------------------------

    def _init_params(self, X):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        k = self.n_components

        idx = rng.choice(n, size=min(n, 20000), replace=False)
        sample = X[idx]
        centres = sample[rng.choice(len(sample), size=k, replace=False)].copy()
        for _ in range(25):
            dist = ((sample[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            assign = dist.argmin(axis=1)
            for j in range(k):
                if np.any(assign == j):
                    centres[j] = sample[assign == j].mean(axis=0)

        self.means_ = centres
        self.covars_ = np.tile(sample.var(axis=0) + self.min_covar, (k, 1))
        self.startprob_ = np.full(k, 1.0 / k)
        trans = np.full((k, k), 1.0 / (k + 4.0))
        np.fill_diagonal(trans, 1.0)
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)

    # -- batched forward / backward -----------------------------------------

    def _forward(self, logb, mask, log_start, log_trans):
        """Scaled forward pass over a padded batch.

        Returns log_alpha (B, T, K) and log_c (B, T), with log_c zero on
        padding so that sums over it are unaffected.
        """
        b, t_max, k = logb.shape
        log_alpha = np.zeros((b, t_max, k))
        log_c = np.zeros((b, t_max))

        la = log_start[None, :] + logb[:, 0]
        c = logsumexp(la, axis=1)
        log_alpha[:, 0] = la - c[:, None]
        log_c[:, 0] = np.where(mask[:, 0], c, 0.0)

        for t in range(1, t_max):
            pred = logsumexp(log_alpha[:, t - 1][:, :, None] + log_trans[None], axis=1)
            la = pred + logb[:, t]
            c = logsumexp(la, axis=1)
            valid = mask[:, t][:, None]
            log_alpha[:, t] = np.where(valid, la - c[:, None], log_alpha[:, t - 1])
            log_c[:, t] = np.where(mask[:, t], c, 0.0)
        return log_alpha, log_c

    def _backward(self, logb, mask, log_trans, log_c):
        b, t_max, k = logb.shape
        log_beta = np.zeros((b, t_max, k))
        for t in range(t_max - 2, -1, -1):
            nxt = logb[:, t + 1] + log_beta[:, t + 1] - log_c[:, t + 1][:, None]
            lb = logsumexp(log_trans[None] + nxt[:, None, :], axis=2)
            log_beta[:, t] = np.where(mask[:, t + 1][:, None], lb, 0.0)
        return log_beta

    # -- public API ---------------------------------------------------------

    def fit(self, X, lengths=None):
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        bounds = self._bounds(X, lengths)
        batches = self._batches(bounds)
        k, d = self.n_components, X.shape[1]
        prev_ll, it = -np.inf, 0

        for it in range(self.n_iter):
            log_start = np.log(self.startprob_ + _LOG_FLOOR)
            log_trans = np.log(self.transmat_ + _LOG_FLOOR)

            acc_start = np.zeros(k)
            acc_trans = np.zeros((k, k))
            acc_gamma = np.zeros(k)
            acc_x = np.zeros((k, d))
            acc_xx = np.zeros((k, d))
            total_ll = 0.0

            for starts, ends in batches:
                Xb, mask = self._pad(X, starts, ends)
                logb = self._log_emission(Xb)
                logb = np.where(mask[:, :, None], logb, 0.0)

                log_alpha, log_c = self._forward(logb, mask, log_start, log_trans)
                log_beta = self._backward(logb, mask, log_trans, log_c)
                total_ll += log_c.sum()

                log_gamma = log_alpha + log_beta
                log_gamma -= logsumexp(log_gamma, axis=2, keepdims=True)
                gamma = np.exp(log_gamma) * mask[:, :, None]

                acc_start += gamma[:, 0].sum(axis=0)
                acc_gamma += gamma.sum(axis=(0, 1))
                # (K, D) accumulations without materialising a (B, T, K, D) tensor
                g2 = gamma.reshape(-1, k)
                x2 = Xb.reshape(-1, d)
                acc_x += g2.T @ x2
                acc_xx += g2.T @ (x2 ** 2)

                # expected transitions, chunked over time
                t_max = logb.shape[1]
                if t_max > 1:
                    step = max(1, self.max_batch_cells // max(len(starts) * k * k, 1))
                    for t0 in range(0, t_max - 1, step):
                        t1 = min(t0 + step, t_max - 1)
                        nxt = (logb[:, t0 + 1:t1 + 1]
                               + log_beta[:, t0 + 1:t1 + 1]
                               - log_c[:, t0 + 1:t1 + 1][:, :, None])
                        log_xi = (log_alpha[:, t0:t1][:, :, :, None]
                                  + log_trans[None, None]
                                  + nxt[:, :, None, :])
                        xi = np.exp(log_xi) * mask[:, t0 + 1:t1 + 1][:, :, None, None]
                        acc_trans += xi.sum(axis=(0, 1))

            self.startprob_ = acc_start / max(acc_start.sum(), _LOG_FLOOR)
            self.transmat_ = acc_trans / np.maximum(
                acc_trans.sum(axis=1, keepdims=True), _LOG_FLOOR
            )
            self.means_ = acc_x / np.maximum(acc_gamma[:, None], _LOG_FLOOR)
            self.covars_ = np.maximum(
                acc_xx / np.maximum(acc_gamma[:, None], _LOG_FLOOR) - self.means_ ** 2,
                self.min_covar,
            )

            if self.verbose:
                print(f"    HMM EM iter {it + 1}: loglik = {total_ll:,.2f}")
            if abs(total_ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = total_ll

        self.monitor_iterations_ = it + 1
        self.loglik_ = total_ll
        return self

    def pointwise_surprisal(self, X, lengths=None):
        """Per-observation score -log p(o_t | o_1..o_{t-1}).

        This is the quantity AETHER uses as the Sequential Expert score. It is
        defined per observation, so unlike a whole-sequence log-likelihood it
        varies across rows and can be ranked. Returned in the row order of X.
        """
        X = np.asarray(X, dtype=float)
        bounds = self._bounds(X, lengths)
        log_start = np.log(self.startprob_ + _LOG_FLOOR)
        log_trans = np.log(self.transmat_ + _LOG_FLOOR)
        out = np.empty(len(X), dtype=float)

        for starts, ends in self._batches(bounds):
            Xb, mask = self._pad(X, starts, ends)
            logb = np.where(mask[:, :, None], self._log_emission(Xb), 0.0)
            _, log_c = self._forward(logb, mask, log_start, log_trans)
            for i, (s, e) in enumerate(zip(starts, ends)):
                out[s:e] = -log_c[i, : e - s]
        return out

    def score(self, X, lengths=None):
        return float(-self.pointwise_surprisal(X, lengths).sum())


# ----------------------------------------------------------------------------
# 2. Exact Shapley values
# ----------------------------------------------------------------------------

def exact_shapley(predict_fn, x_row, background, feature_names=None):
    """Exact interventional Shapley values by full coalition enumeration.

    Returns (phi, base_value) with phi.sum() + base_value == predict_fn(x_row).
    """
    x_row = np.asarray(x_row, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    d = len(x_row)
    cache = {}

    def v(subset):
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        mat = background.copy()
        for j in key:
            mat[:, j] = x_row[j]
        cache[key] = float(np.asarray(predict_fn(mat)).mean())
        return cache[key]

    phi = np.zeros(d)
    for i in range(d):
        rest = [j for j in range(d) if j != i]
        for r in range(len(rest) + 1):
            weight = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
            for combo in itertools.combinations(rest, r):
                phi[i] += weight * (v(list(combo) + [i]) - v(list(combo)))

    return phi, v([])
