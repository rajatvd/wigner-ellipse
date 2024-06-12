# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: wigner-ellipse
#     language: python
#     name: wigner-ellipse
# ---

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from gifify import gifify
from tqdm import tqdm

# make default font size larger
plt.rcParams.update({"font.size": 32})

# make default line width larger
plt.rcParams.update({"axes.linewidth": 4})
plt.rcParams.update({"lines.linewidth": 4})
plt.rcParams.update({"grid.linewidth": 4})

# make default figure size larger
plt.rcParams.update({"figure.figsize": (10, 8)})

# make default markers larger
plt.rcParams.update({"lines.markersize": 20})

# %% [markdown]
# # Sample NxN matrix $J$ with stats:
# $\mathbb{E}[J] = 0$
#
# $\mathbb{E}[J_{ij}^2] = 1/N$
#
# $\mathbb{E}[J_{ij}J_{ji}] = \tau / N, \forall i \neq j$
#


# %%
def sample_matrix(N=50, num_samples=1000, tau=0.5, print_info=True):
    """
    Generate num_samples number of random NxN asymmetric matrix.

    The correlation between J_ij and J_ji is tau.
    The mean of each element is 0.
    The variance of each element is 1.

    Returns:
        J: np.ndarray of shape (N, N, num_samples)
        mean_diff: maximum difference between the mean of J and 0
        std_diff: maximum difference between the std of J and 1
        corr_diff: maximum difference between the correlation of J and tau
        tol: tolerance for the differencek
    """
    # generate a random matrix
    A = np.random.randn(N, N, num_samples) / np.sqrt(N)
    At = np.transpose(A, (1, 0, 2))

    # some quick maffs to get the desired correlation
    a = np.sqrt((1 + tau))
    b = np.sqrt((1 - tau))
    alpha = 0.5 * (a + b)
    beta = 0.5 * (a - b)
    J = alpha * A + beta * At

    # resample the diagonal elements
    # J[np.diag_indices(N)] = np.random.randn(N, num_samples) / np.sqrt(N)
    J[np.diag_indices(N)] = 0

    # compute empirical statistics
    Jt = np.transpose(J, (1, 0, 2))
    means = np.mean(J, axis=2)
    stds = np.mean(J * J, axis=2)
    corrs = np.mean(J * Jt, axis=2)
    expected = (tau + np.eye(N) * (-tau)) / N

    tol = 10 * np.sqrt(1 / num_samples)
    mean_diff = np.max(np.abs(means))
    std_diff = np.max(np.abs(stds - 1.0 / N * (1 - np.eye(N))))
    corr_diff = np.max(np.abs(corrs - expected))

    if print_info:
        print(
            f"mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e}, "
            + f"corr_diff={corr_diff:.2e}, tol={tol:.2e}"
        )

    return J, mean_diff, std_diff, corr_diff, tol


# %%
# check stats
mean_diffs = []
std_diffs = []
corr_diffs = []
tols = []

N = 50
tau = 0.5
num_samples = np.logspace(1, 4, 10).astype(int)
for num_sample in num_samples:
    print(f"Checking num_samples={num_sample}")
    J, mean_diff, std_diff, corr_diff, tol = sample_matrix(
        num_samples=num_sample,
        N=N,
        tau=tau,
    )
    mean_diffs.append(mean_diff)
    std_diffs.append(std_diff)
    corr_diffs.append(corr_diff)
    tols.append(tol)

plt.figure()
plt.grid()
plt.plot(num_samples, mean_diffs, "o-", label="mean")
plt.plot(num_samples, std_diffs, "o-", label="std")
plt.plot(num_samples, corr_diffs, "o-", label="corr")
plt.plot(num_samples, tols, "o-", label="c/sqrt(num_samples)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("num_samples")
plt.ylabel("max difference")
plt.title(f"N={N}, tau={tau}")
# plot legend outside of the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %% [markdown]
# # Compute spectrum and histogram on complex plane
# %%
def spectrum(J):
    """
    Compute the spectrum of the NxN matrix J.

    Returns:
        spectrum: np.ndarray of shape (N, num_samples)
    """
    # compute the eigenvalues
    spectrum = np.linalg.eigvals(J.transpose(2, 0, 1))
    return spectrum


# %%
def plot_spectrum(spectrum, bins=100, tau=0.0, extent=2.0, colorbar=True):
    real = np.real(spectrum).reshape(-1)
    imag = np.imag(spectrum).reshape(-1)

    edges = np.linspace(-extent, extent, bins, endpoint=True)

    H, xedges, yedges = np.histogram2d(real, imag, bins=edges, density=True)

    x = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
    y = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])

    xx, yy = np.meshgrid(x, y)

    # contour plot of the histogram
    plt.contourf(xx, yy, H.T, levels=100, cmap="viridis")
    # plt.xlabel("real")
    # plt.ylabel("imag")
    plt.title(f"Spectrum of J, tau={tau:.2f}, N={N}")
    if colorbar:
        plt.colorbar()

    return H, xedges, yedges


# %%
def plot_ellipse(tau):
    """
    Plot the ellipse with the given tau.
    """
    a = 1 + tau
    b = 1 - tau
    theta = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    plt.plot(x, y, "r", label="ellipse")


# %%
# sample matrix
tau = 0.5
N = 300
num_samples = 100
J = sample_matrix(N=N, num_samples=num_samples, tau=tau)[0]
# %%
# compute spectrum
spec = spectrum(J)
# %%
# plot spectrum
bins = 200
plt.figure()
H, xedges, yedges = plot_spectrum(spec, bins=bins, tau=tau, extent=2.0)
plt.tight_layout()
plt.savefig(f"spectrum_tau={tau}_N={N}.png")
plt.show()

# %%
Ns = [50, 300, 600]
taus = [0.0, 0.5, 0.9, 1.0]

specs = {}
for tau in taus:
    for N in Ns:
        print(f"Calculating spectra tau={tau}, N={N}")
        J = sample_matrix(N=N, num_samples=num_samples, tau=tau)[0]
        spec = spectrum(J)
        specs[(tau, N)] = spec
# %%
fig_dir = "/Users/rajat/Dropbox/Apps/Overleaf/AP229 Final Project/figs/"
extension = "eps"
os.makedirs(f"{fig_dir}/{extension}", exist_ok=True)
plt.subplots(len(Ns), len(taus), figsize=(30, 24))
for n, N in enumerate(Ns):
    for i, tau in enumerate(taus):
        plt.subplot(len(Ns), len(taus), n * len(taus) + i + 1)
        if i == 0:
            plt.ylabel(f"N={N}")
        H, xedges, yedges = plot_spectrum(
            specs[(tau, N)],
            bins=bins,
            tau=tau,
            extent=2.0,
            colorbar=False,
        )
        plot_ellipse(tau)
        if n == 0:
            plt.title(f"tau={tau}")
        else:
            plt.title("")

plt.tight_layout()
plt.savefig(f"{fig_dir}/{extension}/spectra_num_samples={num_samples}.{extension}")
plt.show()

# %% [markdown]
# # Vary $\tau$
# %%
num_samples = 10
N = 300
taus = np.linspace(-0.9, 0.9, 50)
for tau in gifify(tqdm(taus), filename="spectrum.gif"):
    plt.figure()
    J = sample_matrix(N=N, num_samples=num_samples, tau=tau, print_info=False)[0]
    spec = spectrum(J)
    H, xedges, yedges = plot_spectrum(spec, bins=bins, tau=tau, extent=2.0)
    plot_ellipse(tau)

# %%
