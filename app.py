import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import io
import time
import tempfile
import os

try:
    from four_bar import four_bar_kinematics, position_analysis
except ImportError:
    pass

try:
    from slider_crank import slider_crank_kinematics
except ImportError:
    pass

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kinematics Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Custom CSS */
    .stApp { background: #0f1117; }
    
    [data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a3040;
    }
    
    .section-header {
        color: #58a6ff;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-bottom: 1px solid #2a3a55;
        padding-bottom: 6px;
        margin-bottom: 12px;
        margin-top: 16px;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #1f6feb, #0d4a99);
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        padding: 0.5rem;
        margin-bottom: 6px;
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #1f6feb);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(31,111,235,0.4);
    }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox > div > div {
        background: #1e2535 !important;
        color: #e6edf3 !important;
        border: 1px solid #2a3a55 !important;
        border-radius: 6px !important;
    }
    
    h1, h2, h3 { color: #e6edf3 !important; }
    .stRadio label, .stSelectbox label, .stSlider label { color: #8b949e !important; }
    .stInfo { background: #1c2d3f; border-color: #1f6feb; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: build Four-Bar joint positions
# ─────────────────────────────────────────────
def fourbar_joints(L1, L2, L3, L4, theta2_deg, assembly=1):
    """Returns O2, A, B, O4 as (x,y) tuples."""
    t2 = np.radians(theta2_deg)
    try:
        t3d, t4d = position_analysis(L1, L2, L3, L4, theta2_deg, assembly)
        t3 = np.radians(t3d)
        t4 = np.radians(t4d)
        O2 = (0.0, 0.0)
        O4 = (L1, 0.0)
        A  = (L2 * np.cos(t2), L2 * np.sin(t2))
        B  = (O4[0] + L4 * np.cos(t4), O4[1] + L4 * np.sin(t4))
        return O2, A, B, O4, True
    except Exception:
        return None, None, None, None, False


# ─────────────────────────────────────────────
# Helper: build Slider-Crank joint positions
# ─────────────────────────────────────────────
def slider_joints(r, l, theta_deg):
    """Returns O, A, B(slider) as (x,y) tuples."""
    theta = np.radians(theta_deg)
    sin_phi = -r * np.sin(theta) / l
    if abs(sin_phi) > 1.0:
        return None, None, None, False
    phi = np.arcsin(sin_phi)
    O = (0.0, 0.0)
    A = (r * np.cos(theta), r * np.sin(theta))
    B = (r * np.cos(theta) + l * np.cos(phi), 0.0)
    return O, A, B, True


# ─────────────────────────────────────────────
# Animation builders (return GIF bytes)
# ─────────────────────────────────────────────
def make_fourbar_gif(L1, L2, L3, L4, omega2, assembly, fps=30, duration_cycles=2):
    period   = abs(2 * np.pi / omega2) if omega2 != 0 else 2.0
    n_frames = max(30, int(fps * period * duration_cycles))
    n_frames = min(n_frames, 300)        # cap to keep GIF small
    angles   = np.linspace(0, 360 * duration_cycles, n_frames, endpoint=False) % 360

    # Pre-compute valid frames
    frames_data = []
    for ang in angles:
        O2, A, B, O4, ok = fourbar_joints(L1, L2, L3, L4, ang, assembly)
        frames_data.append((ang, O2, A, B, O4, ok))

    # Pick axis limits
    all_x, all_y = [], []
    for _, O2, A, B, O4, ok in frames_data:
        if ok:
            for pt in [O2, A, B, O4]:
                all_x.append(pt[0]); all_y.append(pt[1])
    if not all_x:
        return None
    pad = 0.5
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.grid(True, color="#1e2535", linewidth=0.8)
    for spine in ax.spines.values(): spine.set_color("#2a3a55")
    ax.tick_params(colors="#8b949e")
    ax.set_xlabel("x (m)", color="#8b949e", fontsize=9)
    ax.set_ylabel("y (m)", color="#8b949e", fontsize=9)
    ax.set_title("Four-Bar Mechanism", color="#e6edf3", fontsize=11, fontweight="bold")

    # Legend patches
    legend_elements = [
        mpatches.Patch(color="#3b82f6", label=f"Crank L2={L2}m"),
        mpatches.Patch(color="#f59e0b", label=f"Coupler L3={L3}m"),
        mpatches.Patch(color="#10b981", label=f"Follower L4={L4}m"),
        mpatches.Patch(color="#6b7280", label=f"Ground L1={L1}m"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1e2535", edgecolor="#2a3a55",
              labelcolor="#e6edf3", fontsize=7, loc="upper right")

    line_ground,  = ax.plot([], [], color="#6b7280", lw=2.5, solid_capstyle="round")
    line_crank,   = ax.plot([], [], color="#3b82f6", lw=3,   solid_capstyle="round")
    line_coupler, = ax.plot([], [], color="#f59e0b", lw=3,   solid_capstyle="round")
    line_follower,= ax.plot([], [], color="#10b981", lw=3,   solid_capstyle="round")

    joints_plt,   = ax.plot([], [], "o", color="white", ms=7, zorder=5)
    pivots_plt,   = ax.plot([], [], "^", color="#e55", ms=9, zorder=6)

    angle_text = ax.text(xlim[0]+0.05, ylim[1]-0.2, "", color="#e6edf3", fontsize=9)
    omega_text = ax.text(xlim[0]+0.05, ylim[1]-0.5, f"ω₂ = {omega2} rad/s",
                         color="#8b949e", fontsize=8)

    def _draw(frame_idx):
        ang, O2, A, B, O4, ok = frames_data[frame_idx]
        if not ok:
            return (line_ground, line_crank, line_coupler, line_follower,
                    joints_plt, pivots_plt, angle_text)
        xs = lambda *pts: [p[0] for p in pts]
        ys = lambda *pts: [p[1] for p in pts]

        line_ground.set_data(  xs(O2, O4), ys(O2, O4))
        line_crank.set_data(   xs(O2, A),  ys(O2, A))
        line_coupler.set_data( xs(A,  B),  ys(A,  B))
        line_follower.set_data(xs(B, O4),  ys(B, O4))
        joints_plt.set_data(   xs(A, B),   ys(A, B))
        pivots_plt.set_data(   xs(O2, O4), ys(O2, O4))
        angle_text.set_text(f"θ₂ = {ang:.1f}°")
        return (line_ground, line_crank, line_coupler, line_follower,
                joints_plt, pivots_plt, angle_text)

    ani = animation.FuncAnimation(fig, _draw, frames=n_frames,
                                   interval=1000 // fps, blit=True)
    writer = animation.PillowWriter(fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ani.save(tmp_path, writer=writer)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            gif_bytes = f.read()
    finally:
        os.remove(tmp_path)
    return gif_bytes



def make_slider_gif(r, l, omega, fps=30, duration_cycles=2):
    period   = abs(2 * np.pi / omega) if omega != 0 else 2.0
    n_frames = max(30, int(fps * period * duration_cycles))
    n_frames = min(n_frames, 300)
    angles   = np.linspace(0, 360 * duration_cycles, n_frames, endpoint=False) % 360

    frames_data = []
    for ang in angles:
        O, A, B, ok = slider_joints(r, l, ang)
        frames_data.append((ang, O, A, B, ok))

    all_x = [r + l, -(r + l)]
    all_y = [r, -r]
    pad = 0.3
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - 1.0, max(all_y) + 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.grid(True, color="#1e2535", linewidth=0.8)
    for spine in ax.spines.values(): spine.set_color("#2a3a55")
    ax.tick_params(colors="#8b949e")
    ax.set_xlabel("x (m)", color="#8b949e", fontsize=9)
    ax.set_ylabel("y (m)", color="#8b949e", fontsize=9)
    ax.set_title("Slider-Crank Mechanism", color="#e6edf3", fontsize=11, fontweight="bold")

    # Guide rail
    ax.axhline(0, color="#2a3a55", lw=1.5, linestyle="--")

    legend_elements = [
        mpatches.Patch(color="#3b82f6", label=f"Crank r={r}m"),
        mpatches.Patch(color="#f59e0b", label=f"Connecting rod l={l}m"),
        mpatches.Patch(color="#10b981", label="Slider"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1e2535", edgecolor="#2a3a55",
              labelcolor="#e6edf3", fontsize=7, loc="upper right")

    line_crank,  = ax.plot([], [], color="#3b82f6", lw=3, solid_capstyle="round")
    line_rod,    = ax.plot([], [], color="#f59e0b", lw=3, solid_capstyle="round")
    slider_rect  = plt.Rectangle((0, 0), 0, 0, color="#10b981", zorder=5)
    ax.add_patch(slider_rect)
    pivot_plt,   = ax.plot([], [], "^", color="#e55", ms=9, zorder=6)
    joint_plt,   = ax.plot([], [], "o", color="white", ms=7, zorder=5)
    angle_text   = ax.text(xlim[0]+0.05, ylim[1]-0.2, "", color="#e6edf3", fontsize=9)
    omega_text   = ax.text(xlim[0]+0.05, ylim[1]-0.55, f"ω = {omega} rad/s",
                           color="#8b949e", fontsize=8)

    sw, sh = 0.25, 0.18   # slider width/height

    def _draw(frame_idx):
        ang, O, A, B, ok = frames_data[frame_idx]
        if not ok:
            return (line_crank, line_rod, slider_rect, pivot_plt, joint_plt, angle_text)
        xs = lambda *pts: [p[0] for p in pts]
        ys = lambda *pts: [p[1] for p in pts]

        line_crank.set_data(xs(O, A), ys(O, A))
        line_rod.set_data(  xs(A, B), ys(A, B))
        slider_rect.set_xy((B[0] - sw/2, -sh/2))
        slider_rect.set_width(sw); slider_rect.set_height(sh)
        pivot_plt.set_data([O[0]], [O[1]])
        joint_plt.set_data(xs(A), ys(A))
        angle_text.set_text(f"θ = {ang:.1f}°")
        return (line_crank, line_rod, slider_rect, pivot_plt, joint_plt, angle_text)

    ani = animation.FuncAnimation(fig, _draw, frames=n_frames,
                                   interval=1000 // fps, blit=True)
    writer = animation.PillowWriter(fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ani.save(tmp_path, writer=writer)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            gif_bytes = f.read()
    finally:
        os.remove(tmp_path)
    return gif_bytes



# ─────────────────────────────────────────────
# Sidebar — mechanism selector + generic settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Kinematics Visualizer")
    st.markdown("---")

    mechanism = st.selectbox(
        "Select Mechanism",
        ["Four-Bar Mechanism", "Slider-Crank Mechanism"],
        key="mech_select"
    )

    st.markdown("---")
    
    # We will populate mechanism-specific inputs here inside the condition blocks
    input_container = st.container()

# ─────────────────────────────────────────────
# Top Right Header & Page title
# ─────────────────────────────────────────────
col_img, col_t, col_dn = st.columns([1, 6, 2])
with col_t:
    st.title("🔧 Mechanisms Kinematic Analysis")
    st.markdown("Visualize and analyze **Four-Bar** and **Slider-Crank** mechanisms.")

with col_dn:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📥 Downloads"):
        # Read files for download
        with open(__file__, "rb") as f:
            app_bytes = f.read()
        try:
            with open("four_bar.py", "rb") as f:
                fb_bytes = f.read()
        except:
            fb_bytes = b""
        try:
            with open("slider_crank.py", "rb") as f:
                sc_bytes = f.read()
        except:
            sc_bytes = b""

        st.download_button("⬇ four_bar.py",   data=fb_bytes,   file_name="four_bar.py",   mime="text/plain")
        st.download_button("⬇ slider_crank.py", data=sc_bytes, file_name="slider_crank.py", mime="text/plain")
        st.download_button("⬇ app.py", data=app_bytes, file_name="app.py", mime="text/plain")

st.markdown("---")


# ═════════════════════════════════════════════
#  FOUR-BAR MECHANISM
# ═════════════════════════════════════════════
if mechanism == "Four-Bar Mechanism":

    # ── SIDEBAR PANEL ─────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">🔩 Link Lengths</div>', unsafe_allow_html=True)
        L1 = st.slider("Fixed link L1 (m)", 0.1, 10.0, 2.0, 0.1, key="fb_L1")
        L2 = st.slider("Crank L2 (m)", 0.1, 10.0, 0.5, 0.1, key="fb_L2")
        L3 = st.slider("Coupler L3 (m)", 0.1, 10.0, 2.0, 0.1, key="fb_L3")
        L4 = st.slider("Follower L4 (m)", 0.1, 10.0, 2.0, 0.1, key="fb_L4")

        st.markdown('<div class="section-header" style="margin-top:16px">⚡ Kinematic Parameters</div>', unsafe_allow_html=True)
        omega2 = st.number_input("Crank ω₂ (rad/s)", value=5.0, min_value=0.1, step=0.5, key="fb_w2")
        assembly_choice = st.radio("Assembly Mode", ["Open (+1)", "Crossed (-1)"], key="fb_asm")
        assembly = 1 if "Open" in assembly_choice else -1

        st.markdown('<div class="section-header" style="margin-top:16px">📐 Angle Range</div>', unsafe_allow_html=True)
        start_angle = st.number_input("Start Angle (°)", value=0,   step=5,  key="fb_start")
        end_angle   = st.number_input("End Angle (°)",   value=360, step=5,  key="fb_end")
        step        = st.number_input("Step (°)",        value=5, min_value=1, step=1, key="fb_step")

        st.markdown('<div class="section-header" style="margin-top:16px">🎬 Animation</div>', unsafe_allow_html=True)
        anim_fps    = st.slider("Animation FPS", 5, 30, 15, key="fb_fps")
        anim_cycles = st.slider("Cycles to preview", 1, 5, 2, key="fb_cyc")
        
        # Grashof check
        links = sorted([L1, L2, L3, L4])
        grashof = (links[0] + links[3]) <= (links[1] + links[2])
        if grashof:
            st.success("Grashof: ✓ SATISFIED")
        else:
            st.warning("Grashof: ✗ NOT SATISFIED")

    # Compute kinematic data
    thetas  = np.arange(start_angle, end_angle + 1e-9, step)
    results = []
    errors  = []

    for t in thetas:
        try:
            t3d, t4d, w3, w4, a3, a4 = four_bar_kinematics(
                L1, L2, L3, L4, omega2, t, assembly)
            results.append({
                "θ₂ (°)": t, "θ₃ (°)": t3d, "θ₄ (°)": t4d,
                "ω₃ (rad/s)": w3, "ω₄ (rad/s)": w4,
                "α₃ (rad/s²)": a3, "α₄ (rad/s²)": a4
            })
        except Exception as e:
            errors.append(f"θ₂={t}°: {str(e)}")

    # ── ANIMATION ───────────────────────────────
    st.markdown('<div class="section-header">🎬 Animation</div>', unsafe_allow_html=True)
    with st.spinner("Rendering animation... (this may take a few seconds)"):
        gif_data = make_fourbar_gif(L1, L2, L3, L4, omega2, assembly,
                                    fps=anim_fps, duration_cycles=anim_cycles)
    if gif_data:
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(gif_data, caption="Four-Bar Mechanism Animation", use_container_width=True)
        with col_info:
            st.download_button("⬇ Save animation as GIF", data=gif_data,
                               file_name="four_bar_animation.gif", mime="image/gif")
            # Snapshot
            snap_angle = 45
            O2, A, B, O4, ok = fourbar_joints(L1, L2, L3, L4, snap_angle, assembly)
            if ok:
                fig_s, ax_s = plt.subplots(figsize=(4, 3))
                fig_s.patch.set_facecolor("#0f1117")
                ax_s.set_facecolor("#0f1117")
                for spine in ax_s.spines.values(): spine.set_color("#2a3a55")
                ax_s.tick_params(colors="#8b949e")
                ax_s.grid(True, color="#1e2535", linewidth=0.8)
                ax_s.set_aspect("equal")
                xs = lambda *pts: [p[0] for p in pts]
                ys = lambda *pts: [p[1] for p in pts]
                ax_s.plot(xs(O2, O4), ys(O2, O4), color="#6b7280", lw=2.5, label=f"Ground L1={L1}m")
                ax_s.plot(xs(O2, A),  ys(O2, A),  color="#3b82f6", lw=3.5, label=f"Crank L2={L2}m")
                ax_s.plot(xs(A, B),   ys(A, B),   color="#f59e0b", lw=3.5, label=f"Coupler L3={L3}m")
                ax_s.plot(xs(B, O4),  ys(B, O4),  color="#10b981", lw=3.5, label=f"Follower L4={L4}m")
                ax_s.plot(*zip(O2, O4), "^r", ms=9, zorder=6)
                ax_s.plot(*zip(A, B),   "ow", ms=7, zorder=5)
                ax_s.set_title(f"Snapshot at θ₂={snap_angle}°", color="#e6edf3", fontsize=9)
                ax_s.set_xlabel("x (m)", color="#8b949e", fontsize=7)
                ax_s.set_ylabel("y (m)", color="#8b949e", fontsize=7)
                st.pyplot(fig_s)
                plt.close(fig_s)
    else:
        st.error("Could not render animation. Check link lengths / Grashof condition.")

    # ── PLOTS ────────────────────────────────────
    st.markdown('<div class="section-header">📊 Analysis Graphs</div>', unsafe_allow_html=True)
    if len(results) > 0:
        df = pd.DataFrame(results)
        graph_choice = st.radio("Select variable to plot:", 
                                ["Angles (θ₃, θ₄)", "Velocities (ω₃, ω₄)", "Accelerations (α₃, α₄)"],
                                horizontal=True)
        if "Angles" in graph_choice:
            st.line_chart(df.set_index("θ₂ (°)")[["θ₃ (°)", "θ₄ (°)"]])
        elif "Velocities" in graph_choice:
            st.line_chart(df.set_index("θ₂ (°)")[["ω₃ (rad/s)", "ω₄ (rad/s)"]])
        else:
            st.line_chart(df.set_index("θ₂ (°)")[["α₃ (rad/s²)", "α₄ (rad/s²)"]])
    else:
        st.warning("No valid results — adjust link lengths or angle range.")

    # ── DATA TABLE ───────────────────────────────
    st.markdown('<div class="section-header">📋 Complete Data Table</div>', unsafe_allow_html=True)
    if len(results) > 0:
        df = pd.DataFrame(results)
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Export as CSV", data=csv, file_name="four_bar_data.csv", mime="text/csv")
    else:
        st.warning("No data to display.")

    if errors:
        with st.expander("⚠ Skipped angles (Unreachable geometry)"):
            for e in errors:
                st.warning(e)


# ═════════════════════════════════════════════
#  SLIDER-CRANK MECHANISM
# ═════════════════════════════════════════════
elif mechanism == "Slider-Crank Mechanism":

    # ── SIDEBAR PANEL ─────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">🔩 Link Lengths</div>', unsafe_allow_html=True)
        r = st.slider("Crank length r (m)", 0.05, 5.0, 0.5, 0.05, key="sc_r")
        l = st.slider("Connecting-rod length l (m)", 0.1, 10.0, 2.0, 0.1, key="sc_l")

        st.markdown('<div class="section-header" style="margin-top:16px">⚡ Kinematic Parameters</div>', unsafe_allow_html=True)
        omega = st.number_input("Crank ω (rad/s)", value=5.0, min_value=0.1, step=0.5, key="sc_w")

        st.markdown('<div class="section-header" style="margin-top:16px">📐 Angle Range</div>', unsafe_allow_html=True)
        start_angle = st.number_input("Start Angle (°)", value=0,   step=5,  key="sc_start")
        end_angle   = st.number_input("End Angle (°)",   value=360, step=5,  key="sc_end")
        step        = st.number_input("Step (°)",        value=5, min_value=1, step=1, key="sc_step")

        st.markdown('<div class="section-header" style="margin-top:16px">🎬 Animation</div>', unsafe_allow_html=True)
        anim_fps    = st.slider("Animation FPS", 5, 30, 15, key="sc_fps")
        anim_cycles = st.slider("Cycles to preview", 1, 5, 2, key="sc_cyc")

        # Geometry check
        if r < l:
            st.success("Geometry: ✓ Valid (r < l)")
        else:
            st.warning("Geometry: ⚠ r ≥ l — may cause singularities")

    thetas  = np.arange(start_angle, end_angle + 1e-9, step)
    results = []
    errors  = []

    for t in thetas:
        try:
            phi, w2, a2, vs, a_s = slider_crank_kinematics(r, l, omega, t)
            results.append({
                "θ (°)": t, "φ (°)": phi,
                "ω₂ (rad/s)": w2, "α₂ (rad/s²)": a2,
                "vₛ (m/s)": vs, "aₛ (m/s²)": a_s
            })
        except Exception as e:
            errors.append(f"θ={t}°: {str(e)}")

    # ── ANIMATION ───────────────────────────────
    st.markdown('<div class="section-header">🎬 Animation</div>', unsafe_allow_html=True)
    with st.spinner("Rendering animation... (this may take a few seconds)"):
        gif_data = make_slider_gif(r, l, omega,
                                   fps=anim_fps, duration_cycles=anim_cycles)
    if gif_data:
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(gif_data, caption="Slider-Crank Mechanism Animation", use_container_width=True)
        with col_info:
            st.download_button("⬇ Save animation as GIF", data=gif_data,
                               file_name="slider_crank_animation.gif", mime="image/gif")
            # Snapshot
            snap_angle = 45
            O, A, B, ok = slider_joints(r, l, snap_angle)
            if ok:
                fig_s, ax_s = plt.subplots(figsize=(4, 2.5))
                fig_s.patch.set_facecolor("#0f1117")
                ax_s.set_facecolor("#0f1117")
                for spine in ax_s.spines.values(): spine.set_color("#2a3a55")
                ax_s.tick_params(colors="#8b949e")
                ax_s.grid(True, color="#1e2535", linewidth=0.8)
                ax_s.axhline(0, color="#2a3a55", lw=1.5, linestyle="--")
                xs = lambda *pts: [p[0] for p in pts]
                ys = lambda *pts: [p[1] for p in pts]
                ax_s.plot(xs(O, A), ys(O, A), color="#3b82f6", lw=3.5, label=f"Crank r={r}m")
                ax_s.plot(xs(A, B), ys(A, B), color="#f59e0b", lw=3.5, label=f"Rod l={l}m")
                slider = plt.Rectangle((B[0]-0.12, -0.09), 0.24, 0.18, color="#10b981", label="Slider", zorder=5)
                ax_s.add_patch(slider)
                ax_s.plot([O[0]], [O[1]], "^r", ms=9, zorder=6)
                ax_s.plot([A[0]], [A[1]], "ow", ms=7, zorder=5)
                ax_s.set_aspect("equal")
                ax_s.set_title(f"Snapshot at θ={snap_angle}°", color="#e6edf3", fontsize=9)
                ax_s.set_xlabel("x (m)", color="#8b949e", fontsize=7)
                ax_s.set_ylabel("y (m)", color="#8b949e", fontsize=7)
                st.pyplot(fig_s)
                plt.close(fig_s)
    else:
        st.error("Could not render animation. Check geometry (r must be < l).")

    # ── PLOTS ────────────────────────────────────
    st.markdown('<div class="section-header">📊 Analysis Graphs</div>', unsafe_allow_html=True)
    if len(results) > 0:
        df = pd.DataFrame(results)
        graph_choice = st.radio("Select variable to plot:", 
                                ["Angle & Rod Velocity", "Rod Acceleration", "Slider Vel & Accel"], 
                                horizontal=True)
        if "Velocity" in graph_choice:
            st.line_chart(df.set_index("θ (°)")[["φ (°)", "ω₂ (rad/s)"]])
        elif "Acceleration" in graph_choice and "Rod" in graph_choice:
            st.line_chart(df.set_index("θ (°)")[["α₂ (rad/s²)"]])
        else:
            st.line_chart(df.set_index("θ (°)")[["vₛ (m/s)", "aₛ (m/s²)"]])
    else:
        st.warning("No valid results — adjust parameters.")

    # ── DATA TABLE ───────────────────────────────
    st.markdown('<div class="section-header">📋 Complete Data Table</div>', unsafe_allow_html=True)
    if len(results) > 0:
        df = pd.DataFrame(results)
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Export as CSV", data=csv, file_name="slider_crank_data.csv", mime="text/csv")
    else:
        st.warning("No data to display.")

    if errors:
        with st.expander("⚠ Skipped angles"):
            for e in errors:
                st.warning(e)
