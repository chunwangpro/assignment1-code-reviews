"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import os

import numpy as np
import matplotlib.pyplot as plt


# PART 1 (20 points), Q1: Show computations for the given A and B.
# (XY)_ij = sum_k X_ik Y_kj  (row i of X dotted with column j of Y).
#   AB      = [[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]] = [[3, 2], [2, 3]]
#   BA      = [[1*3+2*2, 1*4+2*1], [0*3+(-1)*2, 0*4+(-1)*1]] = [[7, 6], [-2, -1]]
#   A+B     = [[4, 6], [2, 0]]            ->  (A+B)^T = [[4, 2], [6, 0]]
#   B^T     = [[1, 0], [2, -1]]           ->  AB^T = [[3+8, 0-4], [2+2, 0-1]]
#                                                  = [[11, -4], [4, -1]]
#   trace(A)  = 3 + 1 = 4
#   trace(AB) = 3 + 3 = 6
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    # np.array(..., dtype=float) copies, so the caller's arrays are never touched.
    A = np.array([[3., 4.], [2., 1.]]) if A is None else np.array(A, dtype=float)
    B = np.array([[1., 2.], [0., -1.]]) if B is None else np.array(B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# Characteristic polynomial:
#   det(M - lambda*I) = (2 - lambda)^2 - (-1)(-1) = lambda^2 - 4*lambda + 3
#                     = (lambda - 1)(lambda - 3) = 0   ->  lambda_1 = 1, lambda_2 = 3.
# lambda = 1: (M - I)v = [[1, -1], [-1, 1]] v = 0  ->  v_1 = v_2,  v = [1, 1]/sqrt(2).
# lambda = 3: (M - 3I)v = [[-1, -1], [-1, -1]] v = 0 -> v_1 = -v_2, v = [1, -1]/sqrt(2).
# Verification:
#   M [1, 1]^T  = [2-1, -1+2]^T = [1, 1]^T  = 1 * [1, 1]^T
#   M [1, -1]^T = [2+1, -1-2]^T = [3, -3]^T = 3 * [1, -1]^T
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = np.array([[2., -1.], [-1., 2.]]) if M is None else np.array(M, dtype=float)
    # eigh is the solver for symmetric matrices: real eigenvalues (ascending) and an
    # orthonormal set of eigenvector columns, so a repeated root still gets a full
    # orthonormal basis of its eigenspace.
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Let A be m x n and B be n x m, so AB is m x m and BA is n x n.
#   trace(AB) = sum_{i=1..m} (AB)_ii = sum_{i=1..m} sum_{j=1..n} A_ij B_ji
#   trace(BA) = sum_{j=1..n} (BA)_jj = sum_{j=1..n} sum_{i=1..m} B_ji A_ij
# Both are the same finite double sum over all (i, j) of the scalar products
# A_ij B_ji = B_ji A_ij (real numbers commute), only summed in a different order,
# so trace(AB) = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# A square matrix is invertible iff det != 0:
#   det(C) = 4*6 - 7*2 = 24 - 14 = 10 != 0  ->  C^{-1} exists.
# 2x2 formula: [[a, b], [c, d]]^{-1} = 1/(ad - bc) * [[d, -b], [-c, a]]
#   C^{-1} = (1/10) [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]].
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = np.array([[4., 7.], [2., 6.]]) if C is None else np.array(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# (a) Probabilities sum to 1 and sum_{k>=0} 3^k/k! = e^3 (Taylor series of e^x at x=3):
#       c * e^3 = 1  ->  c = e^{-3} ~= 0.0498.   (X ~ Poisson(3).)
# (b) E[X] = sum_k k e^{-3} 3^k/k! = 3 e^{-3} sum_{k>=1} 3^{k-1}/(k-1)! = 3 e^{-3} e^3 = 3.
#     E[X(X-1)] = sum_k k(k-1) e^{-3} 3^k/k! = 9 e^{-3} sum_{k>=2} 3^{k-2}/(k-2)! = 9.
#     E[X^2] = E[X(X-1)] + E[X] = 12,  Var(X) = E[X^2] - E[X]^2 = 12 - 9 = 3.
# For a general rate r the same steps give c = e^{-r}, E[X] = Var(X) = r.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# (a) Definition of conditional probability: P(A|B) = P(A n B)/P(B), so
#       P(A n B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18.
# (b) P(B|A) = P(A n B)/P(A) = 0.18 / 0.5 = 0.36   (equivalently Bayes' rule).
# (c) Inclusion-exclusion: P(A u B) = P(A) + P(B) - P(A n B) = 0.5 + 0.3 - 0.18 = 0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b
    return p_ab, p_ab / p_a, p_a + p_b - p_ab


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# Flips are i.i.d. Bernoulli(theta): h = 3 heads, n - h = 2 tails.
# (a) L(theta) = theta^3 (1 - theta)^2,  l(theta) = 3 ln(theta) + 2 ln(1 - theta).
#     l'(theta) = 3/theta - 2/(1 - theta) = 0  ->  3(1 - theta) = 2 theta  ->  theta = 3/5.
#     l is strictly concave on (0, 1), so this is the global maximum.
# (b) Posterior ∝ likelihood * prior = theta^3 (1-theta)^2 * 2 theta = 2 theta^4 (1-theta)^2.
#     d/dtheta [4 ln(theta) + 2 ln(1 - theta)] = 4/theta - 2/(1 - theta) = 0
#     ->  4(1 - theta) = 2 theta  ->  theta_MAP = 4/6 = 2/3 ~= 0.667.
# In general MLE = h/n and MAP = (h+1)/(n+1). These also give the boundary maxima:
# h = 0 -> MLE = 0 (L = (1-theta)^n is decreasing); h = n -> MLE = MAP = 1.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    # Posterior ∝ theta^(heads+1) (1-theta)^(tosses-heads): a Beta(heads+2, tosses-heads+1)
    # density whose mode (heads+1)/(tosses+1) is also correct at the boundaries.
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# l(mu, s2) = sum_i ln N(x_i | mu, s2) = -(n/2) ln(2 pi s2) - sum_i (x_i - mu)^2 / (2 s2).
# dl/dmu = sum_i (x_i - mu) / s2 = 0           ->  mu_hat = (1/n) sum_i x_i = 16/4 = 4.
# dl/ds2 = -n/(2 s2) + sum_i (x_i - mu)^2/(2 s2^2) = 0
#                                             ->  s2_hat = (1/n) sum_i (x_i - mu_hat)^2
#        = (9 + 1 + 1 + 9)/4 = 5.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    x = np.array([1., 3., 5., 7.]) if data is None else np.asarray(data, dtype=float).ravel()
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# Likelihood x | mu ~ N(mu, 4), prior mu ~ N(0, 1).
# ln p(mu | x) = -(x - mu)^2 / (2*4) - mu^2 / (2*1) + const.
# d/dmu: (x - mu)/4 - mu = 0  ->  x - mu = 4 mu  ->  mu_MAP = x/5 = 5/5 = 1.
# General form (used in the code): mu_MAP = (tau2 * x + s2 * mu0) / (tau2 + s2),
# here (1*5 + 4*0)/(1 + 4) = 1.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    return float((prior_variance * x + observation_variance * prior_mean)
                 / (prior_variance + observation_variance))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Let mu = E[X] (a constant). By definition Var(X) = E[(X - mu)^2]. Expanding the square
# and using linearity of expectation (E[aX + b] = a E[X] + b):
#   E[(X - mu)^2] = E[X^2 - 2 mu X + mu^2] = E[X^2] - 2 mu E[X] + mu^2
#                 = E[X^2] - 2 mu^2 + mu^2 = E[X^2] - (E[X])^2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# For a random vector X in R^d with mean mu = E[X], the covariance matrix is
#   Sigma = E[(X - mu)(X - mu)^T],  i.e. Sigma_jk = Cov(X_j, X_k)  (symmetric, PSD).
# For a unit vector u (u^T u = 1), the projection Y = u^T X is a scalar with
#   E[Y] = u^T mu  (linearity), so
#   Var(Y) = E[(u^T X - u^T mu)^2] = E[(u^T (X - mu))^2].
# A scalar equals its own transpose, so (u^T (X - mu))^2 = u^T (X - mu)(X - mu)^T u, and
# u is constant, so it moves out of the expectation:
#   Var(u^T X) = u^T E[(X - mu)(X - mu)^T] u = u^T Sigma u.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# generate_data draws Z ~ N(0, I) with np.random.default_rng(seed) and returns
# X = mu + Z L^T, where L is the Cholesky factor of Sigma (Sigma = L L^T), so
# X ~ N(mu, Sigma). The default Sigma = [[3, 1.2], [1.2, 1]] has correlation
# rho = 1.2 / sqrt(3) ~= 0.69; for n = 500 and seed = 0 the sample correlation is
# r ~= 0.70. The scatter plot (Figure 1) is an elongated ellipse tilted up to the
# right: large x1 tends to go with large x2. x1 spreads more than x2 (variance ~2.9
# vs ~1.0), so the long axis lies closer to the x1 axis than to the 45 deg line.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = np.array([2., -1.]) if mean is None else np.array(mean, dtype=float)
    covariance = (np.array([[3., 1.2], [1.2, 1.]]) if covariance is None
                  else np.array(covariance, dtype=float))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, mean.shape[0]))   # Z ~ N(0, I)
    L = np.linalg.cholesky(covariance)                    # Sigma = L L^T
    return mean + z @ L.T                                 # rows x_i = mu + L z_i


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# 1. Centering: Xc = X - mean estimates X - mu, so every projection Xc u has zero mean.
# 2. Covariance: S = Xc^T Xc / (n - 1) is the sample version of Sigma in Q1, so the
#    sample variance of the projections Xc u is exactly u^T S u.
# 3. Eigendecomposition: maximizing u^T S u over unit u (Lagrangian
#    u^T S u - lambda (u^T u - 1)) gives S u = lambda u. np.linalg.eigh returns the
#    orthonormal eigenvectors of the symmetric S, and at an eigenvector u^T S u = lambda.
# 4. Selection: sorting eigenvalues in descending order puts the max-variance
#    direction first (PC1). PC2 is orthogonal to it and has the smallest variance.
#    scores = Xc @ components rotates the data into this basis, where the covariance
#    is diag(lambda_1, lambda_2).
# For the default data, the eigenvalues are lambda ~= [3.462, 0.410], PC1 ~= [0.906, 0.424]
# (about 25 deg above the x1 axis) and PC2 ~= [-0.424, 0.906]. Figure 2 shows PC1 along
# the long axis of the point cloud and PC2 perpendicular to it. Each arrow is
# 2*sqrt(lambda) long, so it reaches the 2-sigma ellipse.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype=float)                  # private copy: X is never mutated
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError('X must have shape (n, d) with n >= 2')
    n = X.shape[0]

    # Step 1: center the data (estimate of X - mu in the derivation).
    mean = X.mean(axis=0)
    centered = X - mean

    # Step 2: sample covariance with divisor n - 1 (estimate of Sigma).
    covariance = centered.T @ centered / (n - 1)

    # Step 3: eigendecomposition of the symmetric matrix S. eigh returns ascending
    # eigenvalues and orthonormal eigenvector columns. It also returns an orthonormal
    # basis for tied eigenvalues (e.g. isotropic data), unlike a generic eig.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Step 4: sort descending so PC1 is the max-variance direction. S is PSD, so any
    # negative eigenvalue is round-off (rank-deficient data) and is clipped to 0.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0., None)
    components = eigenvectors[:, order]

    # Eigenvector signs are arbitrary. Make each column's largest-magnitude entry
    # positive so results and plots are deterministic.
    pivot = components[np.argmax(np.abs(components), axis=0), np.arange(components.shape[1])]
    components = components * np.where(pivot < 0, -1., 1.)

    # Step 5: coordinates of each sample in the principal basis.
    scores = centered @ components
    return {'mean': mean, 'centered': centered, 'covariance': covariance,
            'eigenvalues': eigenvalues, 'components': components, 'scores': scores}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# PC i explains lambda_i / sum_j lambda_j of the total variance trace(S) = sum_j lambda_j.
# Here PC1 ~= 89.4% and PC2 ~= 10.6% (Figure 3). Keeping only PC1 stores one number per
# point instead of two and keeps ~89% of the variance; the cost is the discarded
# eigenvalue, sum_i ||x_i - x_hat_i||^2 / (n - 1) = lambda_2. In general, keeping k
# components loses sum_{j>k} lambda_j, so k is chosen as the smallest value whose
# cumulative fraction reaches a target (e.g. 90-95%).
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# Plot styling. PC1/PC2 keep the same color in every figure; context points are gray.
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a')        # blue (PC1), orange (PC2), aqua
_INK, _INK_2, _MUTED = '#0b0b0b', '#52514e', '#898781'
_STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': '#e1e0d9', 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.labelcolor': _INK, 'axes.titlecolor': _INK,
    'axes.titlesize': 11.5, 'axes.titlelocation': 'left', 'axes.labelsize': 10.5,
    'xtick.color': '#c3c2b7', 'ytick.color': '#c3c2b7',
    'xtick.labelcolor': _INK_2, 'ytick.labelcolor': _INK_2,
    'legend.frameon': False, 'legend.fontsize': 9,
    'font.size': 10, 'mathtext.fontset': 'dejavusans',
}


def _ellipse(mean, components, eigenvalues, k, num=200):
    """Points on the k-sigma contour of the Gaussian with this mean/eigenbasis."""
    t = np.linspace(0., 2. * np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)])                       # (2, num)
    radii = k * np.sqrt(eigenvalues)[:, None]
    return mean + (components @ (radii * circle)).T                 # (num, 2)


def _limits(points, pad=0.08):
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = pad * (hi - lo)
    return (lo[0] - margin[0], hi[0] + margin[0]), (lo[1] - margin[1], hi[1] + margin[1])


def _aspect(xlim, ylim):
    """Height/width ratio of an equal-aspect axes showing these limits."""
    return (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# Figure 1 (Q2): scatter plot of the generated correlated data.
# Figure 2 (Q3): data, 1/2-sigma covariance ellipses, and both principal directions
#                drawn through the sample mean (arrow length 2*sqrt(lambda)).
# Figure 3 (Q4): explained-variance fraction per component plus the cumulative curve.
# Discussion of each figure: Assignment1_PCA_Analysis.pdf.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    X = np.asarray(X, dtype=float)
    mean, comps = result['mean'], result['components']
    lam = result['eigenvalues']
    frac = explained_variance(lam)
    xlim, ylim = _limits(X)
    figures = []

    with plt.rc_context(_STYLE):
        # Figure 1: the generated data and its correlation structure. The figure is
        # sized to the equal-aspect axes so no blank band is left.
        fig, ax = plt.subplots(figsize=(5.2, 4.4 * _aspect(xlim, ylim) + 1.3),
                               constrained_layout=True)
        r = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        ax.scatter(X[:, 0], X[:, 1], s=16, color=_SERIES[0], alpha=0.6,
                   edgecolors='white', linewidths=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f'Figure 1. Correlated 2D Gaussian sample\n'
                     f'$n$ = {len(X)}, sample correlation $r$ = {r:.2f}')
        figures.append(fig)

        # Figure 2: principal directions through the sample mean.
        fig, ax = plt.subplots(figsize=(6, 4.4), constrained_layout=True)
        ax.scatter(X[:, 0], X[:, 1], s=14, color=_MUTED, alpha=0.4, edgecolors='none',
                   label='samples')
        for k in (1, 2):
            e = _ellipse(mean, comps, lam, k)
            ax.plot(e[:, 0], e[:, 1], color=_INK_2, lw=1, alpha=0.7,
                    label=r'1$\sigma$ / 2$\sigma$ covariance ellipses' if k == 1 else None)
        reach = 4. * np.sqrt(lam[0])
        for i in range(comps.shape[1]):
            u = comps[:, i]
            line = mean + np.outer([-reach, reach], u)
            ax.plot(line[:, 0], line[:, 1], color=_SERIES[i], lw=1, alpha=0.45)
            tip = mean + 2. * np.sqrt(lam[i]) * u
            ax.annotate('', xy=tip, xytext=mean,
                        arrowprops=dict(arrowstyle='-|>', color=_SERIES[i], lw=2.4,
                                        mutation_scale=18, shrinkA=0, shrinkB=0))
            ax.annotate(f'PC{i + 1}', xy=tip, xytext=(8, 4 if i == 0 else -14),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        color=_INK)
            ax.plot([], [], color=_SERIES[i], lw=2.4,
                    label=rf'PC{i + 1}: $u_{i + 1}$ = [{u[0]:.3f}, {u[1]:.3f}], '
                          rf'$\lambda_{i + 1}$ = {lam[i]:.3f}')
        ax.scatter(*mean, s=70, color=_INK, edgecolors='white', linewidths=2, zorder=5,
                   label=f'sample mean [{mean[0]:.2f}, {mean[1]:.2f}]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Figure 2. Principal directions through the sample mean\n'
                     r'(arrow length = 2$\sqrt{\lambda_i}$)')
        ax.legend(loc='upper left')
        figures.append(fig)

        # Figure 3: explained-variance fractions and their cumulative sum.
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
        pos = np.arange(1, len(frac) + 1)
        colors = [_SERIES[i % len(_SERIES)] for i in range(len(frac))]
        ax.bar(pos, frac, width=0.42, color=colors)
        cumulative = np.cumsum(frac)
        ax.plot(pos, cumulative, color=_INK_2, lw=2, marker='o', ms=8,
                markeredgecolor='white', markeredgewidth=2, zorder=4)
        for p, f, c in zip(pos, frac, cumulative):
            inside = f > 0.25
            ax.annotate(f'{f:.1%}', xy=(p, f / 2 if inside else f),
                        xytext=(0, 0 if inside else 6), textcoords='offset points',
                        ha='center', va='center' if inside else 'bottom',
                        color='white' if inside else _INK, fontsize=11, fontweight='bold')
            if p > 1:   # at PC1 the cumulative value equals the bar label
                ax.annotate(f'cumulative {c:.1%}', xy=(p, c), xytext=(-12, 8),
                            textcoords='offset points', ha='right', va='bottom', color=_INK_2)
        # Bars are colored by component (same colors as Figure 2), so the legend uses a
        # neutral swatch for them instead of repeating one component's color.
        handles = [plt.Rectangle((0, 0), 1, 1, color=_MUTED),
                   plt.Line2D([], [], color=_INK_2, lw=2, marker='o', ms=7)]
        ax.legend(handles, ['individual fraction (bar per component)', 'cumulative fraction'],
                  loc='center right')
        ax.set_xticks(pos, [rf'PC{i}' + '\n' + rf'$\lambda$ = {v:.3f}'
                            for i, v in zip(pos, lam)])
        ax.set_xlim(0.4, len(frac) + 0.9)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_ylabel('fraction of total variance')
        ax.set_xlabel('principal component')
        ax.set_title(r'Figure 3. Explained variance fraction $\lambda_i\,/\,(\lambda_1 + \lambda_2)$')
        figures.append(fig)

    return figures


if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    np.set_printoptions(precision=4, suppress=True)

    print('=== Part 1 ===')
    names = ('AB', 'BA', '(A+B)^T', 'AB^T', 'trace(A)', 'trace(AB)')
    for name, value in zip(names, matrix_operations()):
        print(f'{name} =\n{value}')
    values, vectors = eigensystem()
    M = np.array([[2., -1.], [-1., 2.]])
    print('eigenvalues of M:', values, '\neigenvectors (columns):\n', vectors)
    for lam_i, v in zip(values, vectors.T):
        print(f'  M v = {M @ v}, lambda v = {lam_i * v}')
    print('C^-1 =\n', matrix_inverse())

    print('\n=== Part 2 ===')
    print('Q1: (c, E[X], Var(X)) =', distribution_moments())
    print('Q2: P(A n B), P(B|A), P(A u B) =', tuple(round(p, 4) for p in event_probabilities()))
    print('Q3: (MLE, MAP) =', coin_estimates())
    print('Q4: (mu_MLE, sigma2_MLE) =', gaussian_mle())
    print('Q5: mu_MAP =', gaussian_map())

    print('\n=== Part 3 ===')
    X = generate_data()
    result = pca(X)
    print(f'sample correlation: {np.corrcoef(X.T)[0, 1]:.4f}')
    print('covariance:\n', result['covariance'])
    print('eigenvalues:', result['eigenvalues'])
    print('components (columns):\n', result['components'])
    print('explained variance:', explained_variance(result['eigenvalues']))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    file_names = ('fig1_correlated_data.png', 'fig2_principal_directions.png',
                  'fig3_explained_variance.png')
    for fig, file_name in zip(make_plots(X, result), file_names):
        fig.savefig(os.path.join(out_dir, file_name), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('saved', os.path.join('figures', file_name))
