"""
experiments.py
==============
Runs every experiment in the Robertson report and writes all figures
(figures/) and numeric tables (results/).

Usage
-----
    python3 experiments.py

Requires NumPy, SciPy and matplotlib.
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# An unstable explicit run is expected to overflow; that divergence is a
# result, not an error, so silence the corresponding warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)

from robertson import (  # noqa: E402
    K1, K2, K3, Y0, T0, T1, TFIRST,
    rhs, jacobian, nonzero_eigenvalues, stiffness_ratio,
    explicit_euler, rk4, heun, backward_euler, trapezoidal,
    adaptive_implicit_euler, newton_residual_trace,
    output_grid, reference_solution, uniform_grid,
    rel_error_in_transient, abs_error_at_t_end, conservation_defect,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def save_csv(name, header, rows):
    path = os.path.join(RES, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {path}")


def fig_save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ==========================================================================
# 0. Reference solution
# ==========================================================================
def run_reference():
    print("[0] reference solution (Radau and BDF at three tolerances)")
    table = []
    Yradau = None
    for rtol in (1e-8, 1e-10, 1e-12):
        t, Yr = reference_solution(rtol=rtol, method="Radau")
        _, Yb = reference_solution(rtol=rtol, method="BDF")
        table.append((rtol, *Yr[-1], *Yb[-1]))
        print(f"    rtol={rtol:g}  Radau y(40)={Yr[-1]}  BDF y(40)={Yb[-1]}")
        Yradau = Yr
    save_csv("reference.csv",
             ["rtol", "y1_radau", "y2_radau", "y3_radau",
              "y1_bdf", "y2_bdf", "y3_bdf"], table)

    print("    conservation defect =", conservation_defect(Yradau))
    print("    min component =", Yradau.min())

    t = output_grid()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t, Yradau[:, 0], label="$y_1$ (A)")
    ax.plot(t, Yradau[:, 1] * 1e5, label="$y_2$ (B) $\\times 10^{5}$")
    ax.plot(t, Yradau[:, 2], label="$y_3$ (C)")
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("concentration")
    ax.set_title("Figure 1: reference solution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_save(fig, "fig1_reference_solution.png")
    return t, Yradau


# ==========================================================================
# 1. Eigenvalues and stiffness ratio — report point 1
# ==========================================================================
def run_eigenvalues(t, Yref):
    print("[1] eigenvalues and stiffness ratio")
    rows = []
    t_plot = []
    fast_plot = []
    slow_plot = []
    S_plot = []
    for tm, y in zip(t, Yref):
        if tm <= 0:
            continue
        S, fast, slow = stiffness_ratio(y)
        if not np.isnan(S):
            t_plot.append(tm)
            fast_plot.append(fast)
            slow_plot.append(slow)
            S_plot.append(S)
    for tm in (1e-8, 1e-4, 1e-2, 1.0, 40.0):
        idx = int(np.argmin(np.abs(t - tm)))
        S, fast, slow = stiffness_ratio(Yref[idx])
        rows.append((tm, f"{fast:.3e}", f"{slow:.3e}",
                     f"{S:.3g}" if not np.isnan(S) else "nan"))
        print(f"    t={tm:g}  |lam_fast|={fast:.3e}  |lam_slow|={slow:.3e}  S={S:.3g}")
    save_csv("eigenvalues_stiffness.csv",
             ["t", "abs_lambda_fast", "abs_lambda_slow", "S"], rows)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
    axes[0].loglog(t_plot, fast_plot, label=r"$|\lambda_{\mathrm{fast}}|$")
    axes[0].loglog(t_plot, slow_plot, label=r"$|\lambda_{\mathrm{slow}}|$")
    axes[0].set_ylabel("eigenvalue magnitude")
    axes[0].set_title("Nonzero Jacobian eigenvalues")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    axes[1].loglog(t_plot, S_plot, "o-", ms=2, lw=1)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("stiffness ratio $S(t)$")
    axes[1].set_title("Stiffness ratio")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_save(fig, "fig2_eigenvalues_stiffness.png")


# ==========================================================================
# 2. Where explicit Euler fails — report point 2
# ==========================================================================
def run_explicit_stability(t, Yref):
    print("[2] explicit Euler stability scan")
    lam_fast_40 = abs(nonzero_eigenvalues(Yref[-1]).real).max()
    h_star = 2.0 / lam_fast_40
    print(f"    |lam_fast(40)| = {lam_fast_40:.4g}  ->  h* = {h_star:.3e}")

    rows = []
    for h in (1e-2, 1e-3, 8e-4, 7e-4, 6e-4, 5.9e-4, 5e-4, 2e-4, 1e-4):
        grid = uniform_grid(h)
        t0 = time.time()
        Yn = explicit_euler(rhs, Y0, grid)
        dt = time.time() - t0
        finite = np.all(np.isfinite(Yn))
        min_y1 = float(np.min(Yn[:, 0])) if finite else np.nan
        min_y2 = float(np.min(Yn[:, 1])) if finite else np.nan
        min_y3 = float(np.min(Yn[:, 2])) if finite else np.nan
        nonnegative = finite and min(min_y1, min_y2, min_y3) >= -1e-12
        div = np.argmax(~np.all(np.isfinite(Yn), axis=1)) if not finite else None
        err = abs_error_at_t_end(Yn, _interp_ref(t, Yref, grid)) if finite else np.nan
        rows.append((h, "yes" if not finite else "no",
                     f"{grid[div]:.3g}" if div is not None else "-",
                     f"{min_y1:.3e}", f"{min_y2:.3e}", f"{min_y3:.3e}",
                     "yes" if nonnegative else "no",
                     f"{err:.2e}" if finite else "-", f"{dt:.2f}"))
        print(f"    h={h:.2e}  diverges={not finite}  "
              f"nonnegative={nonnegative}  min_y1={min_y1:.3e}  "
              f"min_y2={min_y2:.3e}  min_y3={min_y3:.3e}  err={err}")
    save_csv("explicit_euler_stability_scan.csv",
             ["h", "diverges", "divergence_time", "min_y1", "min_y2",
              "min_y3", "nonnegative", "full_state_error_inf_at_40",
              "wall_time_s"], rows)

    # Figure 3: explicit instability and step economy (two panels)
    hs_plot = [1e-2, 1e-3, 8e-4, 7e-4, 6e-4, 5.9e-4, 5e-4, 2e-4, 1e-4]
    stable_h, stable_err = [], []
    unstable_h = []
    for h in hs_plot:
        g = uniform_grid(h)
        Yn = explicit_euler(rhs, Y0, g)
        if np.all(np.isfinite(Yn)):
            stable_h.append(h)
            stable_err.append(abs_error_at_t_end(Yn, _interp_ref(t, Yref, g)))
        else:
            unstable_h.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # left: which step sizes are stable
    axes[0].axvspan(9e-5, h_star, color="seagreen", alpha=0.18)
    axes[0].axvspan(h_star, 2e-2, color="crimson", alpha=0.18)
    axes[0].axvline(h_star, color="crimson", lw=1.4, ls="--")
    axes[0].plot(stable_h, [0.5] * len(stable_h), "o", color="seagreen", ms=7,
                 label="stable")
    axes[0].plot(unstable_h, [0.5] * len(unstable_h), "x", color="crimson", ms=8,
                 label="diverges")
    axes[0].set_xscale("log")
    axes[0].set_xlim(9e-5, 2e-2)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xlabel("$h$")
    axes[0].set_title(f"stability edge  $h^*={h_star:.3g}$", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper center")
    axes[0].grid(alpha=0.3, which="both")

    # right: accuracy of the stable runs
    axes[1].loglog(stable_h, stable_err, "o-", color="steelblue")
    axes[1].set_xlabel("$h$")
    axes[1].set_ylabel("$\\|\\Delta y\\|_\\infty(40)$")
    axes[1].set_title("accuracy of stable explicit runs", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Figure 3: explicit instability and step economy")
    fig_save(fig, "fig3_explicit_instability.png")


def _interp_ref(t, Yref, grid):
    return np.column_stack([np.interp(grid, t, Yref[:, j]) for j in range(3)])


# ==========================================================================
# 3. Conservation, non-negativity, state error — report point 3
# ==========================================================================
def run_diagnostics(t, Yref):
    print("[3] diagnostics")
    rows = []
    for h in (1e-3, 6e-4, 5.9e-4, 2e-4):
        grid = uniform_grid(h)
        Yn = explicit_euler(rhs, Y0, grid)
        finite = np.all(np.isfinite(Yn))
        if finite:
            cons = conservation_defect(Yn)
            min_y1 = np.min(Yn[:, 0])
            min_y2 = np.min(Yn[:, 1])
            min_y3 = np.min(Yn[:, 2])
            ref_on_grid = _interp_ref(t, Yref, grid)
            err = np.max(np.abs(Yn[-1] - ref_on_grid[-1]))
        else:
            cons = np.inf
            min_y1 = min_y2 = min_y3 = np.nan
            err = np.inf
        rows.append((h, f"{cons:.2e}", f"{min_y1:.3e}",
                     f"{min_y2:.3e}", f"{min_y3:.3e}", f"{err:.2e}"))
        print(f"    h={h:.2e}  cons={cons:.2e}  "
              f"min_y1={min_y1:.3e}  min_y2={min_y2:.3e}  "
              f"min_y3={min_y3:.3e}  full_state_err={err:.2e}")
    save_csv("diagnostics.csv",
             ["h", "conservation_defect", "min_y1", "min_y2", "min_y3",
              "full_state_error_inf_at_40"], rows)


# ==========================================================================
# 4. Observed convergence orders
# ==========================================================================
def _uniform_on_window(h, t_end=1e-2):
    n = int(round(t_end / h))
    return np.linspace(0.0, t_end, n + 1)


def run_convergence(t, Yref):
    """
    Observed convergence orders on the initial transient window [0, 1e-2].

    The step sizes are halved four times from h = 1e-4. The reference for each
    run is a tight Radau solve *on that same uniform grid*, so no interpolation
    error contaminates the order estimate.

    Note: at the smallest steps the second-order methods reach ~1e-12 and
    flatten against the reference/round-off floor, so only the first four
    levels are used for the order estimate. That saturation is an artefact,
    not an order defect.
    """
    from scipy.integrate import solve_ivp

    print("[4] observed convergence orders on [0, 1e-2]")
    t_end = 1e-2
    hs = [1e-4 / 2 ** k for k in range(4)]
    res = {}

    for name, fn in (("explicit_euler", lambda g: explicit_euler(rhs, Y0, g)),
                     ("rk4", lambda g: rk4(rhs, Y0, g)),
                     ("heun", lambda g: heun(rhs, Y0, g)),
                     ("backward_euler", lambda g: backward_euler(rhs, jacobian, Y0, g)[0]),
                     ("trapezoidal", lambda g: trapezoidal(rhs, jacobian, Y0, g))):
        errs = []
        for h in hs:
            g = np.linspace(0.0, t_end, int(round(t_end / h)) + 1)
            Yn = fn(g)
            sol = solve_ivp(rhs, (0.0, t_end), Y0, method="Radau",
                            rtol=1e-13, atol=1e-16, t_eval=g)
            errs.append(rel_error_in_transient(Yn, g, sol.y.T, t_end=t_end))
        orders = [np.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
        res[name] = (errs, orders)
        print(f"    {name}: errors={['%.2e' % e for e in errs]} orders={['%.2f' % o for o in orders]}")

    rows = []
    for name, (errs, orders) in res.items():
        rows.append((name, *["%.2e" % e for e in errs], *["%.2f" % o for o in orders]))
    save_csv("convergence.csv",
             ["method", "err_h0", "err_h1", "err_h2", "err_h3",
              "order_0", "order_1", "order_2"], rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, (errs, _) in res.items():
        if name == "explicit_euler":
            # The two Euler errors almost coincide here. Keep their true
            # coordinates, but draw different styles so both remain visible.
            ax.loglog(hs, errs, "o-", color="tab:blue", lw=3,
                      ms=9, mfc="white", mew=2, label=name, zorder=4)
        elif name == "backward_euler":
            ax.loglog(hs, errs, "x--", color="tab:red", lw=1.6,
                      ms=6, mew=1.6, label=name, zorder=5)
        else:
            colors = {"rk4": "tab:orange", "heun": "tab:green",
                      "trapezoidal": "tab:purple"}
            ax.loglog(hs, errs, "o-", color=colors[name], label=name)
    ref = np.array(hs)
    # Anchor each guide line to a representative method error at the same h.
    # This keeps the order comparison readable without changing any data.
    h_anchor = ref[len(ref) // 2]
    anchor_idx = len(ref) // 2
    err_be = res["backward_euler"][0][anchor_idx]
    err_heun = res["heun"][0][anchor_idx]
    err_rk4 = res["rk4"][0][anchor_idx]
    ax.loglog(ref, 1.5 * err_be * (ref / h_anchor), "k--", lw=1,
              label="$O(h)$")
    ax.loglog(ref, 1.8 * err_heun * (ref / h_anchor) ** 2, "k:", lw=1,
              label="$O(h^2)$")
    ax.loglog(ref, 1.8 * err_rk4 * (ref / h_anchor) ** 4, color="0.35",
              ls="-.", lw=1, label="$O(h^4)$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("relative error at $t=10^{-2}$")
    ax.set_title("Figure 5: convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig_save(fig, "fig5_convergence.png")
    return res


# ==========================================================================
# 4b. Adaptive step-size control (step doubling) — required demonstration
# ==========================================================================
def run_adaptive(t, Yref):
    """
    Demonstrate that adaptivity actually cuts the work at matched error.

    The brief is explicit: 'a fixed-h run relabelled adaptive is a fail', so
    we report the real h(t) trajectory, accepted/rejected steps and the error
    against the tight reference, and compare with fixed-step backward Euler.
    """
    print("[4b] adaptive step-size control (step doubling)")

    # fixed-step backward Euler at a step that is stable and reasonably
    # accurate, for the matched-error comparison
    h_fixed = 0.03125
    g_fixed = uniform_grid(h_fixed)
    import time
    t0 = time.time()
    Y_fixed, iters_fixed = backward_euler(rhs, jacobian, Y0, g_fixed)
    dt_fixed = time.time() - t0
    err_fixed = abs_error_at_t_end(Y_fixed, _interp_ref(t, Yref, g_fixed))

    # adaptive, tuned to land in the same error range
    rows = [("fixed BE", h_fixed, len(g_fixed) - 1, 0,
             f"{err_fixed:.2e}", f"{dt_fixed:.2f}")]
    print(f"    fixed BE h={h_fixed:g}: steps={len(g_fixed)-1} "
          f"err_inf={err_fixed:.2e}  t={dt_fixed:.2f}s")

    best = None
    for tol in (1e-6, 1e-7, 1e-8):
        t0 = time.time()
        tA, yA, hA, n_rej = adaptive_implicit_euler(
            rhs, jacobian, Y0, T0, T1, h0=0.5, tol=tol)
        dt = time.time() - t0
        err_inf = float(np.max(np.abs(yA[-1] - Yref[-1])))
        rows.append((f"adaptive BE tol={tol:g}", "-", int(len(hA)), int(n_rej),
                     f"{err_inf:.2e}", f"{dt:.2f}"))
        print(f"    adaptive BE tol={tol:g}: accepted={len(hA)} rejected={n_rej} "
              f"min_h={hA.min():.2e} err_inf={err_inf:.2e} t={dt:.2f}s")
        if best is None or abs(err_inf - err_fixed) < abs(best[1] - err_fixed):
            best = (tol, err_inf, tA, yA, hA, n_rej)

    # trapezoidal is order 2, so its adaptive controller buys much more per
    # step; include it for the matched-error comparison the brief asks for
    t0 = time.time()
    tT, yT, hT, n_rejT = adaptive_implicit_euler(
        rhs, jacobian, Y0, T0, T1, h0=0.5, tol=1e-8, method="trapezoidal")
    dtT = time.time() - t0
    errT = float(np.max(np.abs(yT[-1] - Yref[-1])))
    rows.append(("adaptive TR tol=1e-8", "-", int(len(hT)), int(n_rejT),
                 f"{errT:.2e}", f"{dtT:.2f}"))
    print(f"    adaptive TR tol=1e-8: accepted={len(hT)} rejected={n_rejT} "
          f"min_h={hT.min():.2e} err_inf={errT:.2e} t={dtT:.2f}s")

    save_csv("adaptive.csv",
             ["method", "step_or_tol", "accepted", "rejected",
              "err_inf_at_40", "wall_time_s"], rows)

    tol_b, err_b, tA, yA, hA, n_rej = best
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].semilogx(tA[:-1], hA, marker=".", ms=3, ls="-",
                     label=f"adaptive BE, tol$={tol_b:g}$")
    axes[0].semilogx(tT[:-1], hT, marker=".", ms=3, ls="-",
                     label="adaptive TR, tol$=10^{-8}$")
    axes[0].axhline(h_fixed, color="k", ls=":", lw=1,
                    label=f"fixed $h={h_fixed:g}$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("accepted step $h$")
    axes[0].set_title("adaptive $h(t)$: real, varying step", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(len(hA), err_b, "o", label=f"adaptive BE ({len(hA)} steps)")
    axes[1].loglog(len(hT), errT, "o", label=f"adaptive TR ({len(hT)} steps)")
    axes[1].loglog(len(g_fixed) - 1, err_fixed, "s", color="black",
                   label=f"fixed BE ({len(g_fixed)-1} steps)")
    axes[1].set_xlabel("accepted steps")
    axes[1].set_ylabel("$|\\Delta y_\\infty|(40)$")
    axes[1].set_title("work at matched accuracy", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Figure 7: adaptive step-size control")
    fig_save(fig, "fig7_adaptive.png")


# ==========================================================================
# 5. y2(40) under refinement — report point 4
# ==========================================================================
def run_y2_at_40(t, Yref):
    print("[5] y2(40) under refinement")
    y2_ref = Yref[-1, 1]
    print(f"    reference y2(40) = {y2_ref:.7e}")
    rows = []
    for h in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        g = uniform_grid(h)
        Yb, _ = backward_euler(rhs, jacobian, Y0, g)

        try:
            Yt = trapezoidal(rhs, jacobian, Y0, g)
            tr_y2 = f"{Yt[-1, 1]:.6e}"
            tr_min_y2 = f"{Yt[:, 1].min():.3g}"
            print(f"    h={h:<8g} steps={len(g) - 1:<5d} "
                  f"BE y2(40)={Yb[-1, 1]:.6e} "
                  f"TR y2(40)={tr_y2} TR min y2={tr_min_y2}")
        except RuntimeError as exc:
            tr_y2 = "Newton failed"
            tr_min_y2 = "Newton failed"
            print(f"    h={h:<8g} steps={len(g) - 1:<5d} "
                  f"BE y2(40)={Yb[-1, 1]:.6e} "
                  f"TR failed: {exc}")

        rows.append((h, len(g) - 1, f"{Yb[-1, 1]:.6e}",
                     tr_y2, tr_min_y2))
    # adaptive high-order solvers
    from scipy.integrate import solve_ivp
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        s_r = solve_ivp(rhs, (0, 40), Y0, method="Radau", rtol=rtol, atol=1e-16)
        s_b = solve_ivp(rhs, (0, 40), Y0, method="BDF", rtol=rtol, atol=1e-16)
        rows.append((f"rtol={rtol:g}", "-", f"{s_r.y[1,-1]:.6e}", f"{s_b.y[1,-1]:.6e}", "-"))
        print(f"    rtol={rtol:g}: Radau={s_r.y[1,-1]:.6e}  BDF={s_b.y[1,-1]:.6e}")
    save_csv("y2_at_40.csv",
             ["h_or_rtol", "steps", "backward_euler_y2_40",
              "trapezoidal_y2_40", "trapezoidal_min_y2"], rows)


def run_implicit_y2_tolerances(t, Yref):
    print("[5b] implicit Euler y2(40) at several tolerances")
    y2_ref = Yref[-1, 1]
    rows = []

    for tol in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        tA, yA, hA, n_rej = adaptive_implicit_euler(
            rhs, jacobian, Y0, T0, T1,
            h0=0.5,
            tol=tol,
            method="backward_euler"
        )

        y2_final = yA[-1, 1]
        y2_error = abs(y2_final - y2_ref)

        rows.append((
            f"{tol:.0e}",
            f"{y2_final:.10e}",
            f"{y2_ref:.10e}",
            f"{y2_error:.3e}",
            len(hA),
            n_rej,
            f"{hA.min():.3e}",
            f"{hA.max():.3e}"
        ))

        print(
            f"    tol={tol:.0e}  y2(40)={y2_final:.10e}  "
            f"reference={y2_ref:.10e}  error={y2_error:.3e}  "
            f"accepted={len(hA)}  rejected={n_rej}"
        )

    save_csv(
        "implicit_y2_tolerances.csv",
        ["tolerance", "implicit_y2_40", "reference_y2_40",
         "absolute_error_y2", "accepted_steps", "rejected_steps",
         "min_h", "max_h"],
        rows
    )


# ========================================================================== 
# 6. L-stability comparison
# ==========================================================================
def run_l_stability():
    print("[6] A-stability vs L-stability")
    g = uniform_grid(0.02)
    Yt = trapezoidal(rhs, jacobian, Y0, g)
    Yb, _ = backward_euler(rhs, jacobian, Y0, g)
    print(f"    trapezoidal y2 in [{Yt[:,1].min():.3g}, {Yt[:,1].max():.3g}]")
    print(f"    backward Euler y2 min = {Yb[:,1].min():.3g}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(g[1:], Yt[1:, 1], label="trapezoidal (A-stable, not L-stable)")
    ax.semilogx(g[1:], Yb[1:, 1], label="backward Euler (L-stable)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y_2$")
    ax.set_title("Figure 6: L-stability comparison, $h=0.02$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig_save(fig, "fig6_l_stability.png")


# ==========================================================================
# 6b. Absolute-stability regions with h*lambda overlay — report point 3
# ==========================================================================
def run_stability_regions():
    """
    Compare the absolute-stability regions with the actual h*lambda_j points.

    This is the evidence for the brief's question: is accuracy or stability the
    binding step-size constraint? For Robertson both eigenvalues are real and
    negative, so the binding constraint is the leftmost intercept on the
    negative real axis.
    """
    print("[6b] absolute-stability regions with h*lambda overlay")
    try:
        from stability_regions import stability_region_plot
    except ImportError:
        print("    (stability_regions.py not found, skipping)")
        return
    lam_max = stability_region_plot()
    h_edge = 2.0 / lam_max
    print(f"    binding constraint for explicit Euler: h < {h_edge:.3e} "
          f"(stability), while accuracy needs far smaller h")


# ==========================================================================
# 7. Step economy
# ==========================================================================
def run_economy(t, Yref):
    print("[7] step economy")
    rows = []
    for h in (2e-4, 1e-4):
        g = uniform_grid(h)
        t0 = time.time()
        Yn = explicit_euler(rhs, Y0, g)
        dt = time.time() - t0
        err = abs_error_at_t_end(Yn, _interp_ref(t, Yref, g))
        rows.append(("explicit Euler", f"h={h:g}", len(g) - 1, f"{err:.2e}", f"{dt:.1f}"))
        print(f"    EE h={h:g}: steps={len(g)-1} err={err:.2e} t={dt:.1f}s")
    from scipy.integrate import solve_ivp
    for rtol in (1e-6, 1e-8, 1e-10):
        s = solve_ivp(rhs, (0, 40), Y0, method="Radau", rtol=rtol, atol=1e-16)
        err = abs(Yref[-1, 0] - s.y[0, -1])
        rows.append(("Radau adaptive", f"rtol={rtol:g}", s.t.size, f"{err:.2e}", "-"))
        print(f"    Radau rtol={rtol:g}: steps={s.t.size} err={err:.2e}")
    save_csv("economy.csv",
             ["method", "step_or_tol", "steps", "abs_err_y1_at_40", "wall_time_s"], rows)


# ==========================================================================
def main():
    print("=" * 70)
    print("Robertson experiments")
    print("=" * 70)
    t, Yref = run_reference()
    run_eigenvalues(t, Yref)
    run_explicit_stability(t, Yref)
    run_diagnostics(t, Yref)
    run_convergence(t, Yref)
    run_adaptive(t, Yref)
    run_y2_at_40(t, Yref)
    run_implicit_y2_tolerances(t, Yref)
    run_l_stability()
    run_stability_regions()
    run_economy(t, Yref)
    run_newton_trace()
    print("=" * 70)
    print("done — see figures/ and results/")


def run_newton_trace():
    """
    Print the Newton residual for the first Robertson step.

    Diagnostic for the lightning talk: on the very first step (t = 0 -> h) the
    residual RISES before decaying quadratically, because the initial guess
    y2 = 0 is a degenerate point where the 3e7 y2^2 term is dormant.
    """
    print("[8] Newton residual on the first step (t = 0 -> 0.1)")
    newton_residual_trace(rhs, jacobian, T0, Y0, h=0.1)


if __name__ == "__main__":
    main()
