"""
=============================================================
  SLIDER-CRANK MECHANISM — Kinematic Analysis
=============================================================
  Computes for each crank angle θ:
    • Angular velocity of connecting rod  (ω₂)
    • Angular acceleration of connecting rod (α₂)
    • Velocity of slider  (vₛ)
    • Acceleration of slider  (aₛ)

  Sign convention
  ───────────────
  Crank pivot at origin; slider moves along the x-axis.
  Crank angle θ measured CCW from positive x-axis.
  Connecting-rod angle φ measured CCW from positive x-axis.

  Loop-closure equations
  ──────────────────────
  x:  r·cos(θ) + l·cos(φ) = x_slider
  y:  r·sin(θ) + l·sin(φ) = 0   →  φ = arcsin(−r·sin(θ)/l)

  Velocity (differentiate loop eq., α_crank = 0 assumed)
  ───────────────────────────────────────────────────────
  y:  r·ω·cos(θ) + l·ω₂·cos(φ) = 0
        → ω₂ = −r·ω·cos(θ) / (l·cos(φ))
  x:  vₛ = −r·ω·sin(θ) − l·ω₂·sin(φ)

  Acceleration (differentiate velocity eq., α_crank = 0)
  ───────────────────────────────────────────────────────
  y:  −r·ω²·sin(θ) − l·ω₂²·sin(φ) + l·α₂·cos(φ) = 0
        → α₂ = (r·ω²·sin(θ) + l·ω₂²·sin(φ)) / (l·cos(φ))
  x:  aₛ = −r·ω²·cos(θ) − l·α₂·sin(φ) − l·ω₂²·cos(φ)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# Core kinematic functions
# ─────────────────────────────────────────────

def slider_crank_kinematics(r, l, omega, theta_deg):
    """
    Returns (phi_deg, omega2, alpha2, v_slider, a_slider)
    for a single crank angle theta (degrees).
    """
    theta = np.radians(theta_deg)

    # ── Position ──────────────────────────────
    sin_phi = -r * np.sin(theta) / l
    if abs(sin_phi) > 1.0:
        raise ValueError(
            f"Geometry error at θ={theta_deg}°: r·sin(θ)/l = {sin_phi:.4f} "
            "exceeds 1. Connecting rod too short."
        )
    phi     = np.arcsin(sin_phi)          # connecting-rod angle (rad)
    cos_phi = np.cos(phi)

    # ── Velocity ──────────────────────────────
    omega2   = -r * omega * np.cos(theta) / (l * cos_phi)
    v_slider = -r * omega * np.sin(theta) - l * omega2 * np.sin(phi)

    # ── Acceleration ──────────────────────────
    alpha2   = (r * omega**2 * np.sin(theta) + l * omega2**2 * np.sin(phi)) \
               / (l * cos_phi)
    a_slider = (-r * omega**2 * np.cos(theta)
                - l * alpha2   * np.sin(phi)
                - l * omega2**2 * np.cos(phi))

    return np.degrees(phi), omega2, alpha2, v_slider, a_slider


# ─────────────────────────────────────────────
# Pretty table printer
# ─────────────────────────────────────────────

def print_table(results):
    header = (
        f"{'θ (°)':>8} | {'φ (°)':>8} | "
        f"{'ω₂ (rad/s)':>12} | {'α₂ (rad/s²)':>13} | "
        f"{'vₛ (m/s)':>10} | {'aₛ (m/s²)':>11}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in results:
        theta, phi, w2, a2, vs, as_ = row
        print(
            f"{theta:>8.2f} | {phi:>8.3f} | "
            f"{w2:>12.4f} | {a2:>13.4f} | "
            f"{vs:>10.4f} | {as_:>11.4f}"
        )
    print(sep)


# ─────────────────────────────────────────────
# Plot results
# ─────────────────────────────────────────────

def plot_results(r, l, omega, results):
    arr = np.array(results)
    theta_vals  = arr[:, 0]
    omega2_vals = arr[:, 2]
    alpha2_vals = arr[:, 3]
    vs_vals     = arr[:, 4]
    as_vals     = arr[:, 5]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Slider-Crank Kinematics  |  r={r} m, l={l} m, ω={omega} rad/s",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    def styled_plot(ax, x, y, ylabel, color, title):
        ax.plot(x, y, color=color, linewidth=2)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Crank Angle θ (°)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(theta_vals[0], theta_vals[-1])

    styled_plot(fig.add_subplot(gs[0, 0]), theta_vals, omega2_vals,
                "ω₂ (rad/s)",  "#1f77b4", "Connecting-Rod Angular Velocity")
    styled_plot(fig.add_subplot(gs[0, 1]), theta_vals, alpha2_vals,
                "α₂ (rad/s²)", "#d62728", "Connecting-Rod Angular Acceleration")
    styled_plot(fig.add_subplot(gs[1, 0]), theta_vals, vs_vals,
                "vₛ (m/s)",    "#2ca02c", "Slider Velocity")
    styled_plot(fig.add_subplot(gs[1, 1]), theta_vals, as_vals,
                "aₛ (m/s²)",   "#ff7f0e", "Slider Acceleration")

    plt.savefig("slider_crank_results.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved → slider_crank_results.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("    SLIDER-CRANK MECHANISM — Kinematic Analysis")
    print("=" * 60)
    print("\n  Inputs required:")
    print("    r  — Crank length (m)")
    print("    l  — Connecting-rod length (m)")
    print("    ω  — Crank angular velocity (rad/s)")
    print("    θ  — Crank angle(s)")

    # ── Inputs ────────────────────────────────
    r     = float(input("\n  Crank length r (m)              : "))
    l     = float(input("  Connecting-rod length l (m)     : "))
    omega = float(input("  Crank angular velocity ω (rad/s): "))

    print("\n  Angle unit:")
    print("    1 → Degrees  (default)")
    print("    2 → Radians")
    unit_raw = input("  Choice [1/2]: ").strip()
    use_radians = (unit_raw == "2")
    unit_label  = "rad" if use_radians else "°"

    print(f"\n  Enter crank angles θ ({unit_label}).")
    print("  Options:")
    print("    • Range  →  start end step   (e.g.  0 360 10  or  0 6.28 0.1)")
    print("    • List   →  v1 v2 v3 …       (e.g.  0 90 180 270)")
    raw = input("  Input: ").split()

    if len(raw) == 3:
        try:
            start, end, step = map(float, raw)
            thetas_input = np.arange(start, end + 1e-9, step)
        except ValueError:
            thetas_input = list(map(float, raw))
    else:
        thetas_input = list(map(float, raw))

    # Convert to degrees internally
    if use_radians:
        thetas_deg = [np.degrees(t) for t in thetas_input]
        thetas_display = list(thetas_input)   # keep original for display
    else:
        thetas_deg     = list(thetas_input)
        thetas_display = list(thetas_input)

    # ── Compute ───────────────────────────────
    results, errors = [], []
    for t_deg, t_disp in zip(thetas_deg, thetas_display):
        try:
            phi, w2, a2, vs, as_ = slider_crank_kinematics(r, l, omega, t_deg)
            # Store display angle (original unit) + results
            phi_disp = np.radians(phi) if use_radians else phi
            results.append((t_disp, phi_disp, w2, a2, vs, as_))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        print("\n  ⚠  Geometry warnings:")
        for e in errors:
            print("    ", e)

    if not results:
        print("\n  No valid results to display.")
        return

    # ── Display ───────────────────────────────
    print("\n  RESULTS")
    arr = np.array(results)
    ang_unit = "rad" if use_radians else "°"
    header = (
        f"{'θ ('+ang_unit+')':>10} | {'φ ('+ang_unit+')':>10} | "
        f"{'ω₂ (rad/s)':>12} | {'α₂ (rad/s²)':>13} | "
        f"{'vₛ (m/s)':>10} | {'aₛ (m/s²)':>11}"
    )
    sep = "─" * len(header)
    print(sep); print(header); print(sep)
    for row in results:
        theta, phi, w2, a2, vs, as_ = row
        print(
            f"{theta:>10.4f} | {phi:>10.4f} | "
            f"{w2:>12.4f} | {a2:>13.4f} | "
            f"{vs:>10.4f} | {as_:>11.4f}"
        )
    print(sep)

    print(f"\n  Column key:")
    print(f"    θ  — crank angle (input)                      [{ang_unit}]")
    print(f"    φ  — connecting-rod angle                     [{ang_unit}]")
    print(f"    ω₂ — angular velocity of connecting rod       [rad/s]")
    print(f"    α₂ — angular acceleration of connecting rod   [rad/s²]")
    print(f"    vₛ — velocity of slider                       [m/s]")
    print(f"    aₛ — acceleration of slider                   [m/s²]")

    # ── Plot ──────────────────────────────────
    if len(results) > 1:
        # plot_results uses degree-based thetas internally
        results_deg = []
        if use_radians:
            for row in results:
                results_deg.append((np.degrees(row[0]), np.degrees(row[1]),
                                    row[2], row[3], row[4], row[5]))
        else:
            results_deg = results
        do_plot = input("\n  Generate plots? (y/n): ").strip().lower()
        if do_plot == "y":
            plot_results(r, l, omega, results_deg)


if __name__ == "__main__":
    main()
