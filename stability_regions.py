"""
stability_regions.py
====================
Absolute-stability regions for the one-step methods used on the Robertson
problem, with the actual h*lambda_j points overlaid.

This answers the project brief's requirement: "Compare the absolute-stability
regions with h*lambda_j, where lambda_j are eigenvalues of the Jacobian
evaluated at a stated equilibrium or along the computed trajectory. Determine
from evidence whether accuracy or stability is the binding step-size
constraint."

Based on the Week-2 workshop scaffold rk4_stability_region.py, generalised to
take the Robertson Jacobian and step sizes.

Run with:
    python stability_regions.py
Outputs:  figures/fig4_stability_regions.png
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robertson import rhs, jacobian, nonzero_eigenvalues, reference_solution

# --------------------------------------------------------------------------
# Stability functions R(z), z = h*lambda
# --------------------------------------------------------------------------
def R_explicit_euler(z):
    return 1.0 + z


def R_heun(z):
    """Explicit RK2 (Heun): R(z) = 1 + z + z^2/2."""
    return 1.0 + z + z ** 2 / 2.0


def R_rk4(z):
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def R_implicit_euler(z):
    return 1.0 / (1.0 - z)


def R_trapezoidal(z):
    """A-stable, but R(z) -> -1 as z -> -inf: NOT L-stable."""
    return (1.0 + z / 2.0) / (1.0 - z / 2.0)


METHODS = [
    ("explicit Euler", R_explicit_euler),
    ("Heun (RK2)", R_heun),
    ("RK4", R_rk4),
    ("implicit Euler", R_implicit_euler),
    ("trapezoidal", R_trapezoidal),
]


def draw_region(R_func, ax, label, xlim=(-6, 2), ylim=(-4, 4), nx=600, ny=600):
    """Fill the region |R(z)| <= 1 (the absolute-stability region)."""
    x = np.linspace(xlim[0], xlim[1], num=nx)
    y = np.linspace(ylim[0], ylim[1], num=ny)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    with np.errstate(all="ignore"):
        mag = np.abs(R_func(Z))
    ax.contourf(X, Y, mag, levels=[0, 1], colors=["#bcd4ee"], alpha=0.85)
    ax.contour(X, Y, mag, levels=[1], colors="k", linewidths=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\mathrm{Re}(z)$")
    ax.set_ylabel(r"$\mathrm{Im}(z)$")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.25)


def stability_region_plot(h_values=(5e-4, 5.9e-4, 0.25), save=True):
    """
    Plot every method's stability region with the Robertson h*lambda_j marked.

    The Robertson Jacobian eigenvalues are real and negative, so the binding
    constraint is the leftmost intercept of each region on the negative real
    axis: -2 for explicit Euler and Heun, about -2.785 for RK4, and unbounded
    for the implicit methods.
    """
    t, Yref = reference_solution()

    # Frozen-Jacobian diagnostic at t = 40, not a nonlinear stability proof.
    lam_end = nonzero_eigenvalues(Yref[-1])
    lam_max = float(np.max(np.abs(lam_end.real)))

    n = len(METHODS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, R) in zip(axes, METHODS):
        draw_region(R, ax, name)
        for h in h_values:
            z = h * lam_end
            ax.scatter(z.real, z.imag, s=45, marker="x", c="crimson",
                       zorder=10, linewidths=1.6)
        if R in (R_explicit_euler, R_heun, R_rk4):
            region_edge = -2.0 if R in (R_explicit_euler, R_heun) else -2.785
            h_edge = region_edge / -lam_max
            ax.axvline(region_edge, color="crimson", ls=":", lw=1)
            ax.text(0.03, 0.94, f"$h^*={h_edge:.2e}$", transform=ax.transAxes,
                    fontsize=8, va="top", color="crimson")
        else:
            ax.text(0.03, 0.94, "unbounded on $\\mathrm{Re}(z)<0$",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color="seagreen")

    for ax in axes[len(METHODS):]:
        ax.axis("off")
        ax.text(0.03, 0.9,
                "Overlaid step sizes:\n"
                + "\n".join(f"h = {h:g}" for h in h_values)
                + f"\n\nFor h={max(h_values):g}, the fast point lies\n"
                  f"outside this view (Re(z) about {-max(h_values) * lam_max:.0f}).",
                transform=ax.transAxes, va="top", fontsize=8, wrap=True)

    # one shared legend on the first empty slot
    axes[0].scatter([], [], s=45, marker="x", c="crimson", linewidths=1.6,
                    label=r"$h\lambda_j$ at $t=40$")
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Absolute-stability regions with $h\\lambda_j$ overlay "
        + f"($|\\lambda_{{\\mathrm{{fast}}}}(40)|={lam_max / 1e3:.3f}\\times10^3$)"
    )
    fig.tight_layout()
    if save:
        output_dir = Path(__file__).resolve().parent / "figures"
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "fig4_stability_regions.png", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"lambda_max(40) = {lam_max:.4g}")
    print(f"explicit Euler / Heun stability edge: h* = 2/{lam_max:.4g} "
          f"= {2.0 / lam_max:.3e}")
    print(f"RK4 edge: h* = {2.785 / lam_max:.3e}")
    print("implicit Euler / trapezoidal: no real-axis restriction")
    return lam_max


def main():
    stability_region_plot()


if __name__ == "__main__":
    main()
