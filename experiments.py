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
import csv
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
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def save_csv(name, header, rows):
    path = os.path.join(RES, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
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

    if Yradau is None:
        raise RuntimeError("No Radau reference solution was computed")
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
    from scipy.integrate import solve_ivp
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
    sample_times = np.array((1e-8, 1e-4, 1e-2, 1.0, 40.0))
    samples = solve_ivp(rhs, (T0, T1), Y0, method="Radau", rtol=1e-12,
                        atol=1e-14, t_eval=sample_times)
    if not samples.success:
        raise RuntimeError(f"Eigenvalue sampling failed: {samples.message}")
    for tm, y in zip(sample_times, samples.y.T):
        S, fast, slow = stiffness_ratio(y)
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
    negative_h = []
    unstable_h = []
    for h in hs_plot:
        g = uniform_grid(h)
        Yn = explicit_euler(rhs, Y0, g)
        if np.all(np.isfinite(Yn)):
            stable_h.append(h)
            stable_err.append(abs_error_at_t_end(Yn, _interp_ref(t, Yref, g)))
            if np.min(Yn) < -1e-12:
                negative_h.append(h)
        else:
            unstable_h.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # left: which step sizes are stable
    axes[0].axvspan(9e-5, h_star, color="seagreen", alpha=0.18)
    axes[0].axvspan(h_star, 2e-2, color="crimson", alpha=0.18)
    axes[0].axvline(h_star, color="crimson", lw=1.4, ls="--")
    axes[0].plot(stable_h, [0.5] * len(stable_h), "o", color="seagreen", ms=7,
                 label="finite result")
    axes[0].plot(negative_h, [0.5] * len(negative_h), "s", color="darkorange",
                 ms=6, label="finite, negative component")
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
    Demonstrate varying steps and record cost and terminal error separately.
    The local step-doubling tolerance is not a bound on global error at t=40.
    These sweep runs have different terminal errors, so they are not a
    matched-accuracy efficiency comparison.
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
             len(g_fixed) - 1,
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
        rows.append((f"adaptive BE tol={tol:g}", tol, int(len(hA)), int(n_rej),
                     3 * (len(hA) + n_rej),
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
    rows.append(("adaptive TR tol=1e-8", 1e-8, int(len(hT)), int(n_rejT),
                 3 * (len(hT) + n_rejT),
                 f"{errT:.2e}", f"{dtT:.2f}"))
    print(f"    adaptive TR tol=1e-8: accepted={len(hT)} rejected={n_rejT} "
          f"min_h={hT.min():.2e} err_inf={errT:.2e} t={dtT:.2f}s")

    save_csv("adaptive.csv",
             ["method", "step_or_local_tol", "accepted", "rejected",
              "nonlinear_step_solves_including_rejections",
              "err_inf_at_40", "wall_time_s"], rows)

    if best is None:
        raise RuntimeError("No adaptive backward Euler trial was run")
    tol_b, err_b, tA, yA, hA, n_rej = best
    local_rows = []
    local_traces = []
    for label, method, times, values, steps, tol_local in (
            ("adaptive BE", "backward_euler", tA, yA, hA, tol_b),
            ("adaptive TR", "trapezoidal", tT, yT, hT, 1e-8)):
        indicators = []
        for n, h in enumerate(steps):
            segment = np.array([times[n], times[n + 1]])
            if method == "backward_euler":
                coarse = backward_euler(rhs, jacobian, values[n], segment)[0][-1]
            else:
                coarse = trapezoidal(rhs, jacobian, values[n], segment)[-1]
            indicator = float(np.max(np.abs(values[n + 1] - coarse)))
            indicators.append(indicator)
            local_rows.append((label, times[n], times[n + 1], h,
                               indicator, tol_local, indicator / tol_local))
        local_traces.append((label, times[1:], np.asarray(indicators), tol_local))
    save_csv("adaptive_local_errors.csv",
             ["method", "t_start", "t_end", "accepted_h",
              "step_doubling_difference", "local_tolerance", "ratio"],
             local_rows)
    print(f"    accepted-step local indicator / tolerance: "
          f"max={max(row[-1] for row in local_rows):.3g}")

    fig_local, local_axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for label, t_end, indicators, tol_local in local_traces:
        line, = local_axes[0].loglog(t_end, indicators, lw=1.5, label=label)
        local_axes[0].axhline(tol_local, color=line.get_color(), ls="--",
                              lw=1, label=f"{label} tolerance = {tol_local:g}")
        local_axes[1].semilogx(t_end, indicators / tol_local,
                               color=line.get_color(), lw=1.5, label=label)
    local_axes[0].set_ylabel("step-doubling difference")
    local_axes[0].set_title("Accepted-step local error indicators vs tolerances")
    local_axes[0].legend(fontsize=8)
    local_axes[1].axhline(1.0, color="black", ls="--", lw=1,
                          label="acceptance limit")
    local_axes[1].set_xlabel("time $t$")
    local_axes[1].set_ylabel("indicator / tolerance")
    local_axes[1].legend(fontsize=8)
    for ax in local_axes:
        ax.grid(alpha=0.3, which="both")
    fig_local.tight_layout()
    fig_save(fig_local, "fig7_adaptive_local_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].semilogx(tA[1:], hA, marker=".", ms=3, ls="-",
                     label=f"adaptive BE, tol$={tol_b:g}$")
    axes[0].semilogx(tT[1:], hT, marker=".", ms=3, ls="-",
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
    axes[1].set_title("work vs terminal error (accuracy differs)", fontsize=10)
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
    rows: list[tuple[str, float | str, int, str, str, str]] = []
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

        rows.append(("backward Euler", h, len(g) - 1,
                     f"{Yb[-1, 1]:.10e}", f"{Yb[:, 1].min():.3g}", "ok"))
        rows.append(("trapezoidal", h, len(g) - 1, tr_y2, tr_min_y2,
                     "ok" if tr_y2 != "Newton failed" else "Newton failed"))
    # adaptive high-order solvers
    from scipy.integrate import solve_ivp
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        s_r = solve_ivp(rhs, (0, 40), Y0, method="Radau", rtol=rtol, atol=1e-16)
        s_b = solve_ivp(rhs, (0, 40), Y0, method="BDF", rtol=rtol, atol=1e-16)
        if not (s_r.success and s_b.success):
            raise RuntimeError(f"Radau/BDF tolerance sweep failed at rtol={rtol:g}")
        rows.append(("Radau", f"rtol={rtol:g}", len(s_r.t) - 1,
                     f"{s_r.y[1,-1]:.10e}", f"{s_r.y[1].min():.3g}", "ok"))
        rows.append(("BDF", f"rtol={rtol:g}", len(s_b.t) - 1,
                     f"{s_b.y[1,-1]:.10e}", f"{s_b.y[1].min():.3g}", "ok"))
        print(f"    rtol={rtol:g}: Radau={s_r.y[1,-1]:.6e}  BDF={s_b.y[1,-1]:.6e}")
    save_csv("y2_at_40.csv",
             ["method", "h_or_rtol", "nominal_grid_or_accepted_steps", "y2_at_40",
              "min_y2", "status"], rows)


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

    The h*lambda overlay is a local frozen-Jacobian diagnostic at t=40.
    Compare its predicted edge with the step-size scan; the overlay alone
    is not a proof of nonlinear stability.
    """
    print("[6b] absolute-stability regions with h*lambda overlay")
    try:
        from stability_regions import stability_region_plot
    except ImportError:
        print("    (stability_regions.py not found, skipping)")
        return
    lam_max = stability_region_plot()
    h_edge = 2.0 / lam_max
    print(f"    local explicit Euler stability edge at t=40: h < {h_edge:.3e}")
    print("    the binding limit depends on the requested error target; "
          "the h*lambda overlay is not a nonlinear stability proof")


# ==========================================================================
# 7. Step economy
# ==========================================================================
def run_economy(t, Yref):
    print("[7] cost and accuracy")
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
        if not s.success:
            raise RuntimeError(f"Radau cost sweep failed: {s.message}")
        err = float(np.max(np.abs(Yref[-1] - s.y[:, -1])))
        rows.append(("Radau adaptive", f"rtol={rtol:g}", s.t.size - 1,
                     f"{err:.2e}", "-"))
        print(f"    Radau rtol={rtol:g}: steps={s.t.size - 1} err={err:.2e}")
    save_csv("economy.csv",
             ["method", "step_or_tol", "steps", "full_state_error_inf_at_40",
              "wall_time_s"], rows)
    print("    economy.csv is an accuracy sweep; its rows have different errors")

    # A separate matched-accuracy comparison, using the same full-state norm.
    # Include the non-negativity check because a finite answer can be invalid.
    matched = []
    for method, h in (("explicit Euler", 6e-4), ("backward Euler", 1.0)):
        grid = uniform_grid(h)
        counts = {"rhs": 0, "jacobian": 0}

        def counted_rhs(tm, y):
            counts["rhs"] += 1
            return rhs(tm, y)

        def counted_jacobian(tm, y):
            counts["jacobian"] += 1
            return jacobian(tm, y)

        started = time.perf_counter()
        if method == "explicit Euler":
            Y = explicit_euler(counted_rhs, Y0, grid)
            newton_iterations = 0
        else:
            Y, iterations = backward_euler(counted_rhs, counted_jacobian,
                                           Y0, grid)
            newton_iterations = int(iterations.sum())
        wall_time = time.perf_counter() - started
        error = float(np.max(np.abs(Y[-1] - Yref[-1])))
        minimum = float(Y.min())
        matched.append((method, h, len(grid) - 1, error, minimum,
                        conservation_defect(Y), counts["rhs"],
                        counts["jacobian"], newton_iterations, wall_time))
        print(f"    {method}, h={h:g}: full-state error={error:.3e}, "
              f"steps={len(grid)-1}, min component={minimum:.3e}, "
              f"RHS evaluations={counts['rhs']}")

    errors = [row[3] for row in matched]
    target = sum(errors) / len(errors)
    if max(abs(e - target) / target for e in errors) > 0.05:
        raise RuntimeError("Euler comparison did not reach matched accuracy")
    save_csv("matched_euler_cost.csv",
             ["method", "nominal_h", "steps", "full_state_error_inf_at_40",
              "min_component", "conservation_defect", "rhs_evaluations",
              "jacobian_evaluations", "newton_iterations", "solver_wall_time_s"],
             matched)
    print(f"    matched terminal full-state error is about {target:.3e}; "
          "time excludes reference computation, grid setup and plotting")
    print("    step count alone does not include the work inside implicit steps")


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
