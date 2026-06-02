import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Arrow
import numpy as np
from netCDF4 import Dataset
import os
from scipy.linalg import svd as scipy_svd
from scipy.spatial import ConvexHull

import time
import xarray as xr


def PCHA_multi_init(
    X,
    noc,
    I=None,
    U=None,
    opts=None,
    n_init=10,
    init_method="furthest_sum",
    random_state=None,
    verbose=True
):
    """Run PCHA multiple times with different random initializations
    and keep the solution with minimum SSE.

    Parameters
    ----------
    n_init : int
        Number of random initializations
    init_method : {"furthest_sum", "random_convex"}
        Initialization strategy for C
    random_state : int, optional
        RNG seed for reproducibility

    Returns
    -------
    best_XC, best_S, best_C, best_SSE, best_varexpl
    """
    
    n_feat, n_samp = X.shape
    if I is None:
        I = np.arange(n_samp)
    if U is None:
        U = np.arange(n_samp)

    if random_state is not None:
        np.random.seed(random_state)

    best_SSE = np.inf
    best_result = None
    
    print(" ")
    print("Principal Convex Hull Analysis / Archetypal Analysis")
    print(f"A {noc} component model will be fitted")
    print("To stop algorithm press control C")
    print(" ")

    for k in range(n_init):
        # --------------------------------------------------
        # Build initialization
        # --------------------------------------------------
        opts_k = {} if opts is None else dict(opts)

        if init_method == "furthest_sum":
            if I is None:
                I = np.arange(X.shape[1])

            i = FurthestSum(X[:, I], noc, start_index=np.random.randint(len(I)))
            C0 = np.zeros((len(I), noc))
            C0[i, np.arange(noc)] = 1.0

        elif init_method == "random_convex":
            C0 = np.random.rand(len(I), noc)
            C0 /= np.sum(C0, axis=0, keepdims=True)

        elif init_method == "coreset":
            if I is None:
                I = np.arange(X.shape[1])
            coreset = aa_abs_coreset(X.T, noc, rng=None)
            i = coreset['indices']
            C0 = np.zeros((len(I), noc))
            C0[i, np.arange(noc)] =1.0
            
        else:
            raise ValueError("Unknown init_method")

        opts_k["C"] = C0
        
        # --------------------------------------------------
        # Run PCHA
        # --------------------------------------------------
        XC, S, C, SSE, varexpl = PCHA(X, noc, I=I, U=U, opts=opts_k)

        if verbose:
            print(f"[Init {k+1:02d}/{n_init}] SSE = {SSE:.4e}, VarExpl = {100*varexpl:.2f}%")

        # --------------------------------------------------
        # Keep best solution
        # --------------------------------------------------
        if SSE < best_SSE:
            best_SSE = SSE
            best_result = (XC.copy(), S.copy(), C.copy(), SSE, varexpl)

    return best_result


def FurthestSum(X, noc, start_index=None):
    """FurthestSum initialization for Archetypal Analysis / PCHA

    Parameters
    ----------
    X : ndarray (features × samples)
        Data matrix
    noc : int
        Number of points to select
    start_index : int, optional
        Index of the first selected point

    Returns
    -------
    indices : ndarray (noc,)
        Selected column indices of X
    """

    _, n_samples = X.shape

    if noc > n_samples:
        raise ValueError("noc must be <= number of samples")

    # Choose start index
    if start_index is None:
        start_index = np.random.randint(n_samples)

    indices = np.zeros(noc, dtype=int)
    indices[0] = start_index

    # Precompute squared norms
    X_norm2 = np.sum(X * X, axis=0)

    # Distance accumulator
    dist_sum = np.zeros(n_samples)

    # First distance update
    xi = X[:, start_index]
    dist = X_norm2 - 2 * (X.T @ xi) + np.sum(xi * xi)
    dist_sum += dist

    for k in range(1, noc):
        # Exclude already chosen indices
        dist_sum[indices[:k]] = -np.inf

        # Pick furthest point
        i = np.argmax(dist_sum)
        indices[k] = i

        # Update distance sum
        xi = X[:, i]
        dist = X_norm2 - 2 * (X.T @ xi) + np.sum(xi * xi)
        dist_sum += dist

    return indices


def coreset(X, m):
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C


def aa_abs_coreset(X, m, rng=None):
    """
    Python translation of MATLAB function aa_abs_coreset.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix where rows are observations and columns are dimensions.
    m : int
        Size of the coreset (number of sampled points).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    coreset : dict
        Dictionary with keys:
        - 'q' : sampling probabilities (length n)
        - 'indices' : sampled row indices (length m)
        - 'X': sampled data points (m, d)
        - 'W'  : weights for sampled points (length m)
    """
    
    if rng is None:
        rng = np.random.default_rng()
    
    n = X.shape[0]

    # Mean-center X and compute D = sum((X - mean(X,1)).^2, 2)
    X_mean = X.mean(axis=0)
    D = np.sum((X - X_mean) ** 2, axis=1)

    # q = D / sum(D)
    q = D / np.sum(D)

    # indices = datasample(1:n, m, 'Replace', false, 'Weights', q)
    indices = rng.choice(n, size=m, replace=True, p=q)

    # X(coreset.indices, :)
    Xc = X[indices, :]

    # W = 1 ./ (m * q(indices))
    W = 1.0 / (m * q[indices])

    coreset = {
        "q": q,
        "indices": indices,
        "X": Xc,
        "W": W,
    }
    
    return coreset


def mgetopt(opts, name, default):
    return opts[name] if opts is not None and name in opts else default


def PCHA(X, noc, I=None, U=None, opts=None):
    """
    Principal Convex Hull Analysis / Archetypal Analysis

    Parameters
    ----------
    X : ndarray (features × samples)
    noc : int
        Number of components
    I : array-like, optional
        Indices used for dictionary (columns of X)
    U : array-like, optional
        Indices used for reconstruction (columns of X)
    opts : dict, optional
        Options:
            C, S        initial solutions
            maxiter     (default 500)
            conv_crit   (default 1e-6)
            delta       (default 0)

    Returns
    -------
    XC : ndarray
        Archetypes = X[:, I] @ C
    S : ndarray
        Mixing matrix (noc × len(U))
    C : ndarray
        Coefficient matrix (len(I) × noc)
    SSE : float
        Sum of Squared Errors
    varexpl : float
        Variance explained
    """

    if opts is None:
        opts = {}

    conv_crit = mgetopt(opts, "conv_crit", 1e-6)
    maxiter   = mgetopt(opts, "maxiter", 500)
    delta     = mgetopt(opts, "delta", 0.0)
    verbose   = mgetopt(opts, "verbose_inner", 0.0)

    n_feat, n_samp = X.shape

    if I is None:
        I = np.arange(n_samp)
    if U is None:
        U = np.arange(n_samp)

    X_I = X[:, I]
    X_U = X[:, U]

    SST = np.sum(X_U * X_U)

    # ------------------------------------------------------------------
    # Initialize C
    if "C" in opts:
        C = opts["C"].copy()
    else:
        # FurthestSum is not implemented here – random convex init instead
        print
        C = np.random.rand(len(I), noc)
        C /= np.sum(C, axis=0, keepdims=True)

    XC = X_I @ C

    muS = 1.0
    muC = 1.0
    mualpha = 1.0

    # ------------------------------------------------------------------
    # Initialize S
    if "S" in opts:
        S = opts["S"].copy()
        CtXtXC = XC.T @ XC
        XSt = X_U @ S.T
        SSt = S @ S.T
        SSE = SST - 2 * np.sum(XC * XSt) + np.sum(CtXtXC * SSt)
    else:
        XCtX = XC.T @ X_U
        CtXtXC = XC.T @ XC
        S = -np.log(np.random.rand(noc, len(U)))
        S /= np.sum(S, axis=0, keepdims=True)
        SSt = S @ S.T
        SSE = SST - 2 * np.sum(XCtX * S) + np.sum(CtXtXC * SSt)
        S, SSE, muS, SSt = Supdate(S, XCtX, CtXtXC, muS, SST, SSE, 25)

    # ------------------------------------------------------------------
    # Main loop
    iter = 0
    dSSE = np.inf
    t1 = time.process_time()
    varexpl = (SST - SSE) / SST
    
    # --------------------------------------------------
    # Display algorithm profile
    # --------------------------------------------------

    if verbose:
        print(" ")
        print("Principal Convex Hull Analysis / Archetypal Analysis")
        print(f"A {noc} component model will be fitted")
        print("To stop algorithm press control C")
        print(" ")

        dheader = (
            f"{'Iteration':>12} | {'Expl. var.':>12} | {'Cost func.':>12} | "
            f"{'Delta SSEf.':>12} | {'muC':>12} | {'mualpha':>12} | "
            f"{'muS':>12} | {' Time(s)':>12}"
        )

        dline = (
            "-------------+--------------+--------------+--------------+"
            "--------------+--------------+--------------+--------------+"
        )

        print(dline)
        print(dheader)
        print(dline)
    
    while (abs(dSSE) >= conv_crit * abs(SSE)) and iter < maxiter and varexpl < (1 - 10 * np.finfo(float).eps):
        iter += 1
        told = t1

        if verbose:
            if iter % 100 == 0:
                print(dline)
                print(dheader)
                print(dline)
        
        SSE_old = SSE

        # ----- C update
        XSt = X_U @ S.T
        C, SSE, muC, mualpha, CtXtXC, XC = Cupdate(
            X_I, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, 10
        )

        # ----- S update
        XCtX = XC.T @ X_U
        S, SSE, muS, SSt = Supdate(S, XCtX, CtXtXC, muS, SST, SSE, 10)

        dSSE = SSE_old - SSE
        varexpl = (SST - SSE) / SST
        delta_sse = dSSE / abs(SSE) if SSE != 0 else 0.0
        
        if verbose:
            print(
                f"{iter:12.0f} | {varexpl:12.4f} | {SSE:12.4e} | "
                f"{delta_sse:12.4e} | {muC:12.4e} | {mualpha:12.4e} | "
                f"{muS:12.4e} | {t1 - told:12.4f}"
            )

    # Last iteration results
    if verbose:
        print(dline)

        print(
            f"{iter:12.0f} | {varexpl:12.4f} | {SSE:12.4e} | "
            f"{(dSSE / abs(SSE)):12.4e} | {muC:12.4e} | {mualpha:12.4e} | "
            f"{muS:12.4e} | {t1 - told:12.4f}"
        )
    # ------------------------------------------------------------------
    # Sort components by importance
    ind = np.argsort(np.sum(S, axis=1))[::-1]
    S = S[ind, :]
    C = C[:, ind]
    XC = XC[:, ind]

    return XC, S, C, SSE, varexpl


# ======================================================================
# Helper functions
# ======================================================================

def Supdate(S, XCtX, CtXtXC, muS, SST, SSE, niter):
    noc, J = S.shape
    e = np.ones((noc, 1))

    for _ in range(niter):
        SSE_old = SSE
        g = (CtXtXC @ S - XCtX) / (SST / J)
        g = g - e @ np.sum(g * S, axis=0, keepdims=True)

        Sold = S.copy()
        while True:
            S = Sold - muS * g
            S[S < 0] = 0
            S /= np.sum(S, axis=0, keepdims=True)

            SSt = S @ S.T
            SSE = SST - 2 * np.sum(XCtX * S) + np.sum(CtXtXC * SSt)

            if SSE <= SSE_old * (1 + 1e-9):
                muS *= 1.2
                break
            else:
                muS /= 2

    return S, SSE, muS, SSt


def Cupdate(X, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, niter):
    J, noc = C.shape

    if delta != 0:
        alphaC = np.sum(C, axis=0)
        C = C / alphaC

    e = np.ones((J, 1))
    XtXSt = X.T @ XSt

    for _ in range(niter):
        # ----- C update
        SSE_old = SSE
        g = (X.T @ (XC @ SSt) - XtXSt) / SST
        if delta != 0:
            g *= alphaC

        g = g - e @ np.sum(g * C, axis=0, keepdims=True)

        Cold = C.copy()
        while True:
            C = Cold - muC * g
            C[C < 0] = 0
            C /= np.sum(C, axis=0, keepdims=True)

            Ct = C * alphaC if delta != 0 else C
            XC = X @ Ct
            CtXtXC = XC.T @ XC

            SSE = SST - 2 * np.sum(XC * XSt) + np.sum(CtXtXC * SSt)
            if SSE <= SSE_old * (1 + 1e-9):
                muC *= 1.2
                break
            else:
                muC /= 2

        # ----- alpha update
        if delta != 0:
            SSE_old = SSE
            g = (np.diag(CtXtXC @ SSt) / alphaC - np.sum(C * XtXSt, axis=0)) / (SST * J)

            alphaCold = alphaC.copy()
            while True:
                alphaC = alphaCold - mualpha * g
                alphaC = np.clip(alphaC, 1 - delta, 1 + delta)

                XCt = XC * (alphaC / alphaCold)
                CtXtXC = XCt.T @ XCt
                SSE = SST - 2 * np.sum(XCt * XSt) + np.sum(CtXtXC * SSt)

                if SSE <= SSE_old * (1 + 1e-9):
                    mualpha *= 1.2
                    XC = XCt
                    break
                else:
                    mualpha /= 2

    if delta != 0:
        C *= alphaC

    return C, SSE, muC, mualpha, CtXtXC, XC


def draw_pc(ax, pc_data, times, title):

    ax.plot(times, pc_data, color='black', lw=0.7, alpha=0.8)
    ax.fill_between(times, 0, pc_data, where=(pc_data > 0), facecolor='firebrick', alpha=0.5)
    ax.fill_between(times, 0, pc_data, where=(pc_data < 0), facecolor='dodgerblue', alpha=0.5)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)
    return ax
