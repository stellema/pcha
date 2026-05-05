# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
from netCDF4 import Dataset
import time

# %%
file_path = "/oa-decadal-climate/work/observations/sst/oisst/AVHRR-only/reynolds.oi_v02r01.sst.sfc.19820101-20241231.sea.anom.detrend.nc" # 1982-01-01 to 2024-12-31 daily seasonnaly 
ds = xr.open_dataset(file_path)
ds

# %%
from scipy.linalg import svd as scipy_svd
data = np.squeeze(ds['sst'])
lats = ds['lat']
lons = ds['lon']
time_points = data.shape[0]
lat_points = data.shape[1]
lon_points = data.shape[2]

# Remove NaNs and flatten data matrix
[x,y] = np.meshgrid(lons, lats)
y
y = np.sqrt(np.cos(np.deg2rad(y)))
X = data.values * y
X = X.reshape(time_points, lat_points * lon_points)
X_mask = ~np.isnan(np.sum(X, axis=0))
X = X[:,X_mask]
X = X.T

# Ensure X is consistent, i.e. no NaN is found
print("X shape:", X.shape)
print("NaN count:", np.isnan(X).sum())

# Perform economy SVD using numpy linear algorithm
U, L, Vt = np.linalg.svd(np.float64(X), full_matrices=False)

# SVD sanity check
LL=np.diag(L)
print("Check ||X - U * L * Vt|| ~ eps:", np.allclose(np.float64(X), U@LL@Vt))

exp_var = (L**2) / np.sum(L**2) * 100
print("Numpy linalg SVD")
print(f"Variance explained EOF/PC1: {exp_var[0]:.2f}%")
print(f"Variance explained EOF/PC2: {exp_var[1]:.2f}%")

# %%
print("Total number of modes (100% of variance): ", L.shape[0])
# Compute total variance
total_variance = np.sum(L**2)

# Compute cumulative explained variance
explained_variance_ratio = np.cumsum(L**2) / total_variance

# Find number of EOFs needed to explain 95% variance
num_eofs_95 = np.searchsorted(explained_variance_ratio, 0.95) + 1  

print(f"Number of EOFs needed to explain 95% variance: {num_eofs_95}")
print("Shape of U:", U.shape)
print("Shape of L:", L.shape)
print("Shape of Vt:", Vt.shape)

# %%
# Plot the explained variance ratio for all modes
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, L.shape[0]+1), explained_variance_ratio * 100, marker='o', linestyle='-')
plt.axhline(95, color='r', linestyle='--', label="95% Variance Explained")
plt.axvline(num_eofs_95, color='g', linestyle='--', label=f"{num_eofs_95} EOFs")
plt.xlim(1, L.shape[0]+1)  # Adjust x-axis to show all modes
plt.xlabel("Number of EOFs")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Explained Variance Ratio vs. Number of EOFs")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Compute individual contributions (variance explained by each EOF)
individual_contribution = (L**2) / np.sum(L**2)


# Plot individual contributions of the first 173 EOFs
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1,L.shape[0]+1), individual_contribution * 100, marker='o')  # 1, 61 and :60
plt.axvline(num_eofs_95, color='g', linestyle='--', label=f"{num_eofs_95} EOFs")
plt.grid(True)
plt.xlim(1, num_eofs_95+10)  # Adjust x-axis to show the first 60 EOFs
plt.ylabel('Explained Variance [%]')
plt.xlabel('EOF')
plt.title('Individual Contribution of EOFs')
plt.savefig('individual_contribution.png', dpi=300)
plt.show()

# %%
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
    """
    Run PCHA multiple times with different random initializations
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
            C0[i, np.arange(noc)] = 1.0
            
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

# %%
def FurthestSum(X, noc, start_index=None):
    """
    FurthestSum initialization for Archetypal Analysis / PCHA

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

# %%
def coreset(X, m):
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C

# %%
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
        - 'q'       : sampling probabilities (length n)
        - 'indices' : sampled row indices (length m)
        - 'X'       : sampled data points (m, d)
        - 'W'       : weights for sampled points (length m)
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

# %%
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

# %%
X_r = LL@Vt
print("PCA/SVD-reduced dataset: X_r", X_r.shape," =  X_r (space, time)")

# %%
# Implement further truncation on PCA/SVD reduced dataset X_r
n_trunc = X_r.shape[0] # 100% of variance retained
# n_trunc = 2          # First 2 PCs
Y = X_r[:n_trunc,:]
print("Truncated PCA/SVD-reduced dataset: Y", Y.shape," =  Y (space, time)")

# %%
def draw_pc(ax, pc_data, title):
    times = ds.time.values
    ax.plot(times, pc_data, color='black', lw=0.7, alpha=0.8)
    ax.fill_between(times, 0, pc_data, where=(pc_data > 0), facecolor='firebrick', alpha=0.5)
    ax.fill_between(times, 0, pc_data, where=(pc_data < 0), facecolor='dodgerblue', alpha=0.5)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)
    
pc1 = Y[0,:] # Y = Y [space, time]
pc2 = Y[1,:]

fig = plt.figure(figsize=(16, 12))
spec = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.3)

ax_pc1 = fig.add_subplot(spec[0, 0])
draw_pc(ax_pc1, pc1, 'PC1 Time Series')
ax_pc2 = fig.add_subplot(spec[0, 1])
draw_pc(ax_pc2, pc2, 'PC2 Time Series')

fig.suptitle('Detrended SST anomalies (whole globe)', fontsize=18, y=0.96)
plt.show()


# %%
noc = 8
opts = {
    "maxiter": 3000,
    "conv_crit": 1e-8,
    "delta": 0.0,
    "verbose_inner": False
}

# XC, S, C, SSE, varexpl = PCHA(X, noc, opts=opts)

XC, S, C, SSE, varexpl = PCHA_multi_init(
    Y,                     # Truncated datset
    noc,                   # Number of archetypes
    opts=opts,             # Inner loop options
    n_init=1000,           # Outer loop number of restart/trials
    init_method="coreset", # Initialisation strategy
    random_state=42        # Random generator intilisation 
)

print(f"SSE: {SSE:.4e}")
print(f"Variance explained: {100*varexpl:.2f}%")
print("XC shape:", XC.shape)
print("S shape:", S.shape)
print("C shape:", C.shape)

# %%
# Stochastic matrices sanity check
print("Sanity check on stochastic matrix S, is np.sum(S.sum(axis=0)) = S.shape[1]? : ", np.allclose(np.sum(S.sum(axis=0)), S.shape[1]))
print("Sanity check on stochastic matrix C, is np.sum(C.sum(axis=0)) = C.shape[1]? : ", np.allclose(np.sum(C.sum(axis=0)), C.shape[1]))

# %%
# Convert S from numpy.matrix to numpy.ndarray
S = np.asarray(S)  # ! bitn

# Create a time array (9630 days)
t = ds.time.values

# Plot time series for each archetype
use_line_collection=True                                # To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
plt.figure(figsize=(10, 10))                            # Adjust figure size
for i in range(S.shape[0]):                             # Loop through each archetype (row)
    plt.subplot(int(noc/2), 2, i + 1)                            # Create subplots (4 rows, 2 columns)
    plt.stem(t, S[i, :], linefmt='b-', markerfmt='bo')  # Plot time series for archetype i
    plt.title(f'Archetype {i+1}')
    plt.xlabel('Time (Seasons)')
    plt.ylabel('S-matrix values')
    plt.ylim(0, 1)
    plt.xlim(left=t[0], right=t[S.shape[1]-1])

plt.tight_layout()                                      # Adjust layout to prevent overlap
plt.show()

# %%
# Plot time series for each archetype
plt.figure(figsize=(10, 10))                            # Adjust figure size
for i in range(C.shape[1]):                             # Loop through each archetype (column)  #C.shape[1]
    plt.subplot(int(noc/2), 2, i + 1)                            # Create subplots (4 rows, 2 columns)
    plt.stem(t, C[:, i], linefmt='r-', markerfmt='ro')  # Plot time series for archetype i
    plt.title(f'Archetype {i+1}')
    plt.xlabel('Time (Seasons)')
    plt.ylabel('C-matrix values')
    plt.ylim(0, np.max(C))
    plt.xlim(left=t[0], right=t[S.shape[1]-1])
    
plt.tight_layout()                                      # Adjust layout to prevent overlap
plt.show()

# %%
from scipy.spatial import ConvexHull
from matplotlib.patches import Arrow

X2 = Y[:2,:]
print("shape of X is", X2.shape)

xc_marksize = 15
xcs_marksize = 7

# Plot all reduced data points
fig, axc = plt.subplots(figsize=(10, 10))
axc.plot(X2[0, :], X2[1, :], 'o', color='k',zorder=1, markersize=xcs_marksize, markerfacecolor=[0.8, 0.8, 0.8])
axc.set_xlabel(r'$\lambda_{1} PC_{1}$')
axc.set_ylabel(r'$\lambda_{2} PC_{2}$')

# Plot convex hull
hull = ConvexHull(X2.T)  # Convex hull of X (shape: (9815, 2))
k = hull.vertices
axc.plot(X2[0, k], X2[1, k], '-k', linewidth=2) 
axc.plot([X2[0, k[-1]], X2[0, k[0]]], [X2[1, k[-1]], X2[1, k[0]]], '-k', linewidth=3)

# Plot archetypes XC
n_arch = noc
c_colors = plt.cm.Paired(np.linspace(0, 1, n_arch))  # Use a color map for archetypes
a_colortype = 'mix'  # Coloring strategy
for i in range(n_arch):
    c = c_colors[i]
    #print(f"Archetype {i + 1}: X-coordinate: {XC[0, i]}, Y-coordinate: {XC[1, i]}")
    axc.plot(XC[0, i], XC[1, i], 'o', color='k', markersize=xc_marksize, markerfacecolor=c)


# PCHA of the projections
XC_2d = XC[:2, :]
XC_2d = np.asarray(XC_2d)
ii = ConvexHull(XC_2d.T).vertices
axc.plot (XC_2d[0, ii],     XC_2d[1, ii], '-', color='#00FF00', linewidth=3) # green
axc.plot([XC_2d[0, ii[-1]], XC_2d[0, ii[0]]], [XC_2d[1, ii[-1]], XC_2d[1, ii[0]]], '-', color='#00FF00', linewidth=3)

# Plot archetypal representation XCS
XCS = np.dot(XC, S)
n_tim=XCS.shape[1]
for i in range(n_tim):  # step 2
    xcs = [XCS[0, i], XCS[1, i]]
    x = [Y[0, i], Y[1, i]]
    
    
    if a_colortype == 'mix':
        ic = np.argmax(S[:, i])
        c = c_colors[ic]
    
    axc.plot(
        xcs[0], xcs[1], 'o',
        color='k',
        zorder=1,
        linewidth=0.5,
        markeredgecolor=c,
        markerfacecolor=c,
        markersize=xcs_marksize
    )
    axc.plot([x[0], xcs[0]], [x[1], xcs[1]], color='k', zorder=0, linewidth=0.5)

# Archetypes mean composites
XS   = np.dot(X2, S.T)
Ssum = np.sum(S, axis=1)
Ssum = Ssum.flatten()
XS   = XS/Ssum

for i in range(n_arch):
    c = c_colors[i]
    axc.plot(XS[0, i], XS[1, i], 's', color='k', zorder=2, markersize=xc_marksize, markerfacecolor=c)
    
for j in range(n_arch):       # Index of the archetype and simplex point
    arrow = Arrow(
        XC[0, j],             # Starting x-coordinate (archetype)
        XC[1, j],             # Starting y-coordinate (archetype)
        XS[0, j] - XC[0, j],  # dx (x-component of arrow direction)
        XS[1, j] - XC[1, j],  # dy (y-component of arrow direction)
        width=20,             # Width of the arrow
        color='k',            # Color of the arrow (black)
        zorder=3
    )
    axc.add_patch(arrow)
    dx= XS[0, j] - XC[0, j]
    dy= XS[1, j] - XC[1, j]
    length = np.sqrt(dx**2 + dy**2)
    print("Length of the arrow:", length)

# Plot axes
axc.plot([-np.max(np.abs(X2)), np.max(np.abs(X2))], [0, 0], '--k')
axc.plot([0, 0], [-np.max(np.abs(X2)), np.max(np.abs(X2))], '--k')
axc.set_aspect('equal') 

plt.show()

# %%
A_map = data.values[:noc,:,:]

# %%
A_map = A_map.reshape(noc, lat_points * lon_points)

# %%
A_map[:,X_mask] = np.transpose(U[:,:Y.shape[0]]@XC)

# %%
A_map = A_map.reshape(noc, lat_points, lon_points)

# %%
A_map.shape

# %%
xc_marksize  = 15
xcs_marksize = 10

# Set up the figure
fig = plt.figure(figsize=(40, 35))

# Define a grid for the layout
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

# Plot archetypal patterns
archetype_order = list(range(noc)) 

# Define positions for the 8 surrounding plots in a 3x3 grid
if noc == 4:
    positions = [
        (0, 0),         (0, 2),  # Top row
                                 # Middle row
        (2, 0),         (2, 2)   # Bottom row
    ]
if noc == 8:
    positions = [
    (0, 0), (0, 1), (0, 2),  # Top row
    (1, 0),         (1, 2),  # Middle row
    (2, 0), (2, 1), (2, 2)   # Bottom row
    ]

# Plot the archetypes around the central graph
clon=210
for idx, (row, col) in enumerate(positions):
    # Get the archetype index based on the new order
    archetype_idx = archetype_order[idx]
    
    # Create the subplot
    ax=fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree(central_longitude=clon))  
    data=np.squeeze(A_map[archetype_idx,:,:])
    vlim=np.nanpercentile(np.abs(data), 99.5)
    cs=ax.pcolormesh(lons, lats, data,
        vmin=-vlim, vmax=vlim,
        transform = ccrs.PlateCarree(central_longitude=0),
        cmap='coolwarm')
    ax.set_extent((-180, 180, -90, 90))
    
    # Title each subplot with the name of the model
    # A_str=str(archetype_idx+1)
    # ax.set_title(f"A{archetype_idx+1}")

    # Draw the coastines for each subplot
    ax.stock_img()
    ax.coastlines()
    c = c_colors[archetype_idx]
    ax.text(0.015, 0.03, f'A{archetype_idx + 1}', transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor=c, alpha=0.8, edgecolor=c, boxstyle='round,pad=0.5'))
    
# Plot convex hull as the central figure        
axc=fig.add_subplot(gs[1,1])
axc.plot(X2[0, :], X2[1, :], 'o', color='k',zorder=1, markersize=xcs_marksize, markerfacecolor=[0.8, 0.8, 0.8])
axc.set_xlabel(r'$\lambda_{1} PC_{1}$', fontsize=14)
axc.set_ylabel(r'$\lambda_{2} PC_{2}$', fontsize=14)

# Plot convex hull
hull = ConvexHull(X2.T)  # Convex hull of X (shape: (9815, 2))
k = hull.vertices
axc.plot(X2[0, k], X2[1, k], '-k', linewidth=2) 
axc.plot([X2[0, k[-1]], X2[0, k[0]]], [X2[1, k[-1]], X2[1, k[0]]], '-k', linewidth=3)

# Plot archetypes XC
n_arch = noc
c_colors = plt.cm.Paired(np.linspace(0, 1, n_arch))  # Use a color map for archetypes
a_colortype = 'mix'  # Coloring strategy
for i in range(n_arch):
    c = c_colors[i]
    #print(f"Archetype {i + 1}: X-coordinate: {XC[0, i]}, Y-coordinate: {XC[1, i]}")
    axc.plot(XC[0, i], XC[1, i], 'o', color='k', markersize=xc_marksize, markerfacecolor=c)


# PCHA of the projections
XC_2d = XC[:2, :]
XC_2d = np.asarray(XC_2d)
ii = ConvexHull(XC_2d.T).vertices
axc.plot (XC_2d[0, ii],     XC_2d[1, ii], '-', color='#00FF00', linewidth=3) # green
axc.plot([XC_2d[0, ii[-1]], XC_2d[0, ii[0]]], [XC_2d[1, ii[-1]], XC_2d[1, ii[0]]], '-', color='#00FF00', linewidth=3)

# Plot archetypal representation XCS
XCS = np.dot(XC, S)
n_tim=XCS.shape[1]
for i in range(n_tim):  # step 2
    xcs = [XCS[0, i], XCS[1, i]]
    x = [Y[0, i], Y[1, i]]
    
    
    if a_colortype == 'mix':
        ic = np.argmax(S[:, i])
        c = c_colors[ic]
    
    axc.plot(
        xcs[0], xcs[1], 'o',
        color='k',
        zorder=1,
        linewidth=0.5,
        markeredgecolor=c,
        markerfacecolor=c,
        markersize=xcs_marksize
    )
    axc.plot([x[0], xcs[0]], [x[1], xcs[1]], color='k', zorder=0, linewidth=0.5)

# Archetypes mean composites
XS   = np.dot(X2, S.T)
Ssum = np.sum(S, axis=1)
Ssum = Ssum.flatten()
XS   = XS/Ssum

for i in range(n_arch):
    c = c_colors[i]
    axc.plot(XS[0, i], XS[1, i], 's', color='k', zorder=2, markersize=xc_marksize, markerfacecolor=c)
    
for j in range(n_arch):       # Index of the archetype and simplex point
    arrow = Arrow(
        XC[0, j],             # Starting x-coordinate (archetype)
        XC[1, j],             # Starting y-coordinate (archetype)
        XS[0, j] - XC[0, j],  # dx (x-component of arrow direction)
        XS[1, j] - XC[1, j],  # dy (y-component of arrow direction)
        width=20,             # Width of the arrow
        color='k',            # Color of the arrow (black)
        zorder=3
    )
    axc.add_patch(arrow)
    dx= XS[0, j] - XC[0, j]
    dy= XS[1, j] - XC[1, j]
    length = np.sqrt(dx**2 + dy**2)
    print("Length of the arrow:", length)

# Plot axes
axc.plot([-np.max(np.abs(X2)), np.max(np.abs(X2))], [0, 0], '--k')
axc.plot([0, 0], [-np.max(np.abs(X2)), np.max(np.abs(X2))], '--k')

# Change all spines
for axis in ['top','bottom','left','right']:
    axc.spines[axis].set_linewidth(2)
    
plt.tight_layout()
plt.show()

# %%



