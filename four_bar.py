"""
=============================================================
  FOUR-BAR MECHANISM — Kinematic Analysis
=============================================================
  Link numbering (Grashof convention):
    L1 — fixed link (ground)
    L2 — crank  (input link, driven at ω₂)
    L3 — coupler
    L4 — follower (rocker / output link)

  Loop-closure equation (vector loop):
    L2·e^(iθ₂) + L3·e^(iθ₃) − L4·e^(iθ₄) − L1 = 0

  Position analysis — Freudenstein's method
  ──────────────────────────────────────────
    K1 = L1/L2,  K2 = L1/L4,  K3 = (L2²−L3²+L4²+L1²)/(2·L2·L4)
    A = cos(θ₂) − K1 − K2·cos(θ₂) + K3
    B = −2·sin(θ₂)
    C = K1 − (K2+1)·cos(θ₂) + K3
    θ₄ = 2·atan2(−B ± √(B²−4AC), 2A)   (two assembly modes)

    K4 = L1/L3,  K5 = (L4²−L1²−L2²−L3²)/(2·L2·L3)
    D = cos(θ₂) − K1 − K4·cos(θ₂) + K5
    E = −2·sin(θ₂)
    F = K1 − (K4−1)·cos(θ₂) + K5
    θ₃ = 2·atan2(−E ± √(E²−4DF), 2D)

  Velocity analysis (differentiate loop eq.)
  ───────────────────────────────────────────
    −L2·ω₂·sin(θ₂) − L3·ω₃·sin(θ₃) + L4·ω₄·sin(θ₄) = 0
     L2·ω₂·cos(θ₂) + L3·ω₃·cos(θ₃) − L4·ω₄·cos(θ₄) = 0

    Solve [ω₃, ω₄] via 2×2 system.

  Acceleration analysis
  ──────────────────────
    L2·ω₂²·cos(θ₂) + L3·ω₃²·cos(θ₃) + L3·α₃·sin(θ₃)
      − L4·ω₄²·cos(θ₄) − L4·α₄·sin(θ₄) = 0   (x-component)
    L2·ω₂²·sin(θ₂) + L3·ω₃²·sin(θ₃) − L3·α₃·cos(θ₃)
      − L4·ω₄²·sin(θ₄) + L4·α₄·cos(θ₄) = 0   (y-component)

    Solve [α₃, α₄] via 2×2 system.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys


# ─────────────────────────────────────────────
# Position analysis (Freudenstein)
# ─────────────────────────────────────────────

def position_analysis(L1, L2, L3, L4, theta2_deg, assembly=+1):
    """
    Solve for θ₃, θ₄ (degrees).
    assembly = +1 (open) or -1 (crossed).
    Raises ValueError if mechanism cannot reach position.
    """
    t2 = np.radians(theta2_deg)

    # ── θ₄ ────────────────────────────────────
    K1 = L1 / L2
    K2 = L1 / L4
    K3 = (L2**2 - L3**2 + L4**2 + L1**2) / (2 * L2 * L4)

    A = np.cos(t2) - K1 - K2 * np.cos(t2) + K3
    B = -2 * np.sin(t2)
    C = K1 - (K2 + 1) * np.cos(t2) + K3
    disc4 = B**2 - 4 * A * C
    if disc4 < 0:
        raise ValueError(
            f"θ₂={theta2_deg}°: no real solution for θ₄ "
            "(mechanism cannot reach this position)."
        )
    theta4 = 2 * np.arctan2(-B + assembly * np.sqrt(disc4), 2 * A)

    # ── θ₃ ────────────────────────────────────
    K4 = L1 / L3
    K5 = (L4**2 - L1**2 - L2**2 - L3**2) / (2 * L2 * L3)

    D = np.cos(t2) - K1 - K4 * np.cos(t2) + K5
    E = -2 * np.sin(t2)
    F = K1 - (K4 - 1) * np.cos(t2) + K5
    disc3 = E**2 - 4 * D * F
    if disc3 < 0:
        raise ValueError(
            f"θ₂={theta2_deg}°: no real solution for θ₃."
        )
    theta3 = 2 * np.arctan2(-E + assembly * np.sqrt(disc3), 2 * D)

    return np.degrees(theta3), np.degrees(theta4)


# ─────────────────────────────────────────────
# Velocity analysis
# ─────────────────────────────────────────────

def velocity_analysis(L2, L3, L4, theta2, theta3, theta4, omega2):
    """All angles in radians. Returns (ω₃, ω₄) in rad/s."""
    # [ -L3·sin(θ₃)   L4·sin(θ₄) ] [ω₃]   [ L2·ω₂·sin(θ₂) ]
    # [  L3·cos(θ₃)  -L4·cos(θ₄) ] [ω₄] = [-L2·ω₂·cos(θ₂) ]
    A = np.array([
        [-L3 * np.sin(theta3),  L4 * np.sin(theta4)],
        [ L3 * np.cos(theta3), -L4 * np.cos(theta4)]
    ])
    b = np.array([
        L2 * omega2 * np.sin(theta2),
       -L2 * omega2 * np.cos(theta2)
    ])
    sol = np.linalg.solve(A, b)
    return sol[0], sol[1]    # ω₃, ω₄


# ─────────────────────────────────────────────
# Acceleration analysis
# ─────────────────────────────────────────────

def acceleration_analysis(L2, L3, L4, theta2, theta3, theta4,
                           omega2, omega3, omega4, alpha2=0.0):
    """All angles in radians. Returns (α₃, α₄) in rad/s²."""
    # [  L3·sin(θ₃)  -L4·sin(θ₄) ] [α₃]
    # [ -L3·cos(θ₃)   L4·cos(θ₄) ] [α₄]
    #   = rhs (centripetal + crank acceleration terms)
    A = np.array([
        [ L3 * np.sin(theta3), -L4 * np.sin(theta4)],
        [-L3 * np.cos(theta3),  L4 * np.cos(theta4)]
    ])
    rhs_x = (L2 * omega2**2 * np.cos(theta2)
              + L3 * omega3**2 * np.cos(theta3)
              - L4 * omega4**2 * np.cos(theta4)
              + L2 * alpha2  * np.sin(theta2))
    rhs_y = (L2 * omega2**2 * np.sin(theta2)
              + L3 * omega3**2 * np.sin(theta3)
              - L4 * omega4**2 * np.sin(theta4)
              - L2 * alpha2  * np.cos(theta2))
    b = np.array([rhs_x, rhs_y])
    sol = np.linalg.solve(A, b)
    return sol[0], sol[1]    # α₃, α₄


# ─────────────────────────────────────────────
# Full kinematics for one angle
# ─────────────────────────────────────────────

def four_bar_kinematics(L1, L2, L3, L4, omega2, theta2_deg, assembly=+1):
    t2 = np.radians(theta2_deg)
    t3d, t4d = position_analysis(L1, L2, L3, L4, theta2_deg, assembly)
    t3 = np.radians(t3d)
    t4 = np.radians(t4d)

    omega3, omega4 = velocity_analysis(L2, L3, L4, t2, t3, t4, omega2)
    alpha3, alpha4 = acceleration_analysis(L2, L3, L4, t2, t3, t4,
                                           omega2, omega3, omega4)
    return t3d, t4d, omega3, omega4, alpha3, alpha4


# ─────────────────────────────────────────────
# Pretty table
# ─────────────────────────────────────────────

def print_table(results):
    hdr = (
        f"{'θ₂ (°)':>8} | {'θ₃ (°)':>8} | {'θ₄ (°)':>8} | "
        f"{'ω₃ (rad/s)':>11} | {'ω₄ (rad/s)':>11} | "
        f"{'α₃ (rad/s²)':>12} | {'α₄ (rad/s²)':>12}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for row in results:
        t2, t3, t4, w3, w4, a3, a4 = row
        print(
            f"{t2:>8.2f} | {t3:>8.3f} | {t4:>8.3f} | "
            f"{w3:>11.4f} | {w4:>11.4f} | "
            f"{a3:>12.4f} | {a4:>12.4f}"
        )
    print(sep)


# ─────────────────────────────────────────────
# Plot results
# ─────────────────────────────────────────────

def plot_results(L1, L2, L3, L4, omega2, results):
    arr = np.array(results)
    t2_v  = arr[:, 0]
    w3_v, w4_v = arr[:, 3], arr[:, 4]
    a3_v, a4_v = arr[:, 5], arr[:, 6]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Four-Bar Kinematics  |  L1={L1}, L2={L2}, L3={L3}, L4={L4} m,"
        f"  ω₂={omega2} rad/s",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    def sp(ax, x, ys, labels, colors, ylabel, title):
        for y, lbl, col in zip(ys, labels, colors):
            ax.plot(x, y, color=col, linewidth=2, label=lbl)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Crank Angle θ₂ (°)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(t2_v[0], t2_v[-1])

    sp(fig.add_subplot(gs[0, 0]), t2_v,
       [w3_v], ["ω₃"], ["#1f77b4"],
       "ω₃ (rad/s)", "Coupler Angular Velocity")

    sp(fig.add_subplot(gs[0, 1]), t2_v,
       [w4_v], ["ω₄"], ["#d62728"],
       "ω₄ (rad/s)", "Follower Angular Velocity")

    sp(fig.add_subplot(gs[1, 0]), t2_v,
       [a3_v], ["α₃"], ["#2ca02c"],
       "α₃ (rad/s²)", "Coupler Angular Acceleration")

    sp(fig.add_subplot(gs[1, 1]), t2_v,
       [a4_v], ["α₄"], ["#ff7f0e"],
       "α₄ (rad/s²)", "Follower Angular Acceleration")

    plt.savefig("four_bar_results.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved → four_bar_results.png")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("    FOUR-BAR MECHANISM — Kinematic Analysis")
    print("=" * 65)
    print("\n  Inputs required:")
    print("    L1 — Fixed link length (ground)   [m]")
    print("    L2 — Crank length                 [m]")
    print("    L3 — Coupler length               [m]")
    print("    L4 — Follower length              [m]")
    print("    ω₂ — Crank angular velocity       [rad/s]")
    print("    θ₂ — Crank angle(s)")

    L1 = float(input("\n  Fixed link length   L1 (m): "))
    L2 = float(input("  Crank length        L2 (m): "))
    L3 = float(input("  Coupler length      L3 (m): "))
    L4 = float(input("  Follower length     L4 (m): "))
    omega2 = float(input("  Crank angular velocity ω₂ (rad/s): "))

    print("\n  Assembly mode:")
    print("    1 → Open configuration  (default)")
    print("    2 → Crossed configuration")
    mode_raw = input("  Choice [1/2]: ").strip()
    assembly = -1 if mode_raw == "2" else +1

    print("\n  Angle unit:")
    print("    1 → Degrees  (default)")
    print("    2 → Radians")
    unit_raw = input("  Choice [1/2]: ").strip()
    use_radians = (unit_raw == "2")
    unit_label  = "rad" if use_radians else "°"

    print(f"\n  Enter crank angles θ₂ ({unit_label}).")
    print("  Options:")
    print("    • Range  →  start end step   (e.g.  0 360 5  or  0 6.28 0.1)")
    print("    • List   →  v1 v2 v3 …       (e.g.  30 60 90 120)")
    raw = input("  Input: ").split()

    if len(raw) == 3:
        try:
            start, end, step = map(float, raw)
            thetas_input = np.arange(start, end + 1e-9, step)
        except ValueError:
            thetas_input = list(map(float, raw))
    else:
        thetas_input = list(map(float, raw))

    # Convert to degrees for internal computation
    if use_radians:
        thetas_deg     = [np.degrees(t) for t in thetas_input]
        thetas_display = list(thetas_input)
    else:
        thetas_deg     = list(thetas_input)
        thetas_display = list(thetas_input)

    # ── Compute ───────────────────────────────
    results, errors = [], []
    for t_deg, t_disp in zip(thetas_deg, thetas_display):
        try:
            t3, t4, w3, w4, a3, a4 = four_bar_kinematics(
                L1, L2, L3, L4, omega2, t_deg, assembly
            )
            results.append((t_disp, t3, t4, w3, w4, a3, a4))
        except (ValueError, np.linalg.LinAlgError) as e:
            errors.append(f"θ₂={t_disp} {unit_label}: {e}")

    if errors:
        print("\n  ⚠  Warnings (skipped angles):")
        for e in errors:
            print("    ", e)

    if not results:
        print("\n  No valid results — check link lengths / Grashof condition.")
        return

    # ── Grashof check ─────────────────────────
    links = sorted([L1, L2, L3, L4])
    grashof = (links[0] + links[3]) <= (links[1] + links[2])
    print(f"\n  Grashof condition: {'SATISFIED ✓' if grashof else 'NOT satisfied ✗'}")
    if not grashof:
        print("  (Mechanism is a non-Grashof linkage — limited rotation possible)")

    # ── Display ───────────────────────────────
    print("\n  RESULTS")
    ang_unit = unit_label
    hdr = (
        f"{'θ₂ ('+ang_unit+')':>10} | {'θ₃ (°)':>8} | {'θ₄ (°)':>8} | "
        f"{'ω₃ (rad/s)':>12} | {'ω₄ (rad/s)':>12} | "
        f"{'α₃ (rad/s²)':>13} | {'α₄ (rad/s²)':>13}"
    )
    sep = "─" * len(hdr)
    print(sep); print(hdr); print(sep)
    for row in results:
        t2, t3, t4, w3, w4, a3, a4 = row
        print(
            f"{t2:>10.4f} | {t3:>8.3f} | {t4:>8.3f} | "
            f"{w3:>12.4f} | {w4:>12.4f} | "
            f"{a3:>13.4f} | {a4:>13.4f}"
        )
    print(sep)

    print(f"\n  Column key (OUTPUTS):")
    print(f"    θ₂ — crank angle (input)                          [{ang_unit}]")
    print(f"    θ₃ — coupler angle (position)                     [°]")
    print(f"    θ₄ — follower angle (position)                    [°]")
    print(f"    ω₃ — angular velocity of coupler link             [rad/s]")
    print(f"    ω₄ — angular velocity of follower link            [rad/s]")
    print(f"    α₃ — angular acceleration of coupler link         [rad/s²]")
    print(f"    α₄ — angular acceleration of follower link        [rad/s²]")

    # ── Plot ──────────────────────────────────
    if len(results) > 1:
        do_plot = input("\n  Generate plots? (y/n): ").strip().lower()
        if do_plot == "y":
            # plot_results needs (t2_deg, t3, t4, w3, w4, a3, a4)
            results_for_plot = []
            for row in results:
                t2_disp, t3, t4, w3, w4, a3, a4 = row
                t2_deg = np.degrees(t2_disp) if use_radians else t2_disp
                results_for_plot.append((t2_deg, t3, t4, w3, w4, a3, a4))
            plot_results(L1, L2, L3, L4, omega2, results_for_plot)


if __name__ == "__main__":
    main()

