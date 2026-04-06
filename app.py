import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from four_bar import four_bar_kinematics
except ImportError:
    pass

try:
    from slider_crank import slider_crank_kinematics
except ImportError:
    pass

st.set_page_config(page_title="Kinematics Visualizer", layout="wide")

st.title("Mechanisms Kinematic Analysis Visualizer")
st.markdown("Build and visualize kinematic properties of Four-Bar and Slider-Crank mechanisms.")

mechanism = st.sidebar.selectbox("Select Mechanism", ["Four-Bar Mechanism", "Slider-Crank Mechanism"])

if mechanism == "Four-Bar Mechanism":
    st.header("Four-Bar Mechanism")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Link Lengths")
        L1 = st.number_input("Fixed link L1 (m)", value=2.0, min_value=0.1)
        L2 = st.number_input("Crank L2 (m)", value=0.5, min_value=0.1)
        L3 = st.number_input("Coupler L3 (m)", value=2.0, min_value=0.1)
        L4 = st.number_input("Follower L4 (m)", value=2.0, min_value=0.1)
        
        st.subheader("Kinematic Parameters")
        omega2 = st.number_input("Crank angular velocity ω₂ (rad/s)", value=10.0)
        assembly_choice = st.radio("Assembly Mode", ["Open (+1)", "Crossed (-1)"])
        assembly = 1 if "Open" in assembly_choice else -1
        
        start_angle = st.number_input("Start Angle (°)", value=0)
        end_angle = st.number_input("End Angle (°)", value=360)
        step = st.number_input("Step (°)", value=5, min_value=1)
        
    with col2:
        thetas = np.arange(start_angle, end_angle + 1e-9, step)
        results = []
        errors = []
        
        for t in thetas:
            try:
                # Returns: t3d, t4d, omega3, omega4, alpha3, alpha4
                t3d, t4d, w3, w4, a3, a4 = four_bar_kinematics(L1, L2, L3, L4, omega2, t, assembly)
                results.append({
                    "θ₂ (°)": t,
                    "θ₃ (°)": t3d,
                    "θ₄ (°)": t4d,
                    "ω₃ (rad/s)": w3,
                    "ω₄ (rad/s)": w4,
                    "α₃ (rad/s²)": a3,
                    "α₄ (rad/s²)": a4
                })
            except Exception as e:
                errors.append(f"θ₂={t}°: {str(e)}")
                
        if len(results) > 0:
            df = pd.DataFrame(results)
            st.write("### Kinematic Data")
            st.dataframe(df.style.format("{:.3f}"), width='stretch')
            
            st.write("### Plots")
            
            tab1, tab2, tab3 = st.tabs(["Angles (θ₃, θ₄)", "Angular Velocities (ω₃, ω₄)", "Angular Accelerations (α₃, α₄)"])
            
            with tab1:
                st.line_chart(df.set_index("θ₂ (°)")[["θ₃ (°)", "θ₄ (°)"]])
            
            with tab2:
                st.line_chart(df.set_index("θ₂ (°)")[["ω₃ (rad/s)", "ω₄ (rad/s)"]])
                
            with tab3:
                st.line_chart(df.set_index("θ₂ (°)")[["α₃ (rad/s²)", "α₄ (rad/s²)"]])
                
            links = sorted([L1, L2, L3, L4])
            grashof = (links[0] + links[3]) <= (links[1] + links[2])
            if grashof:
                st.success("Grashof condition: SATISFIED ✓ (Continuous relative rotation is possible)")
            else:
                st.warning("Grashof condition: NOT SATISFIED ✗ (Mechanism is a non-Grashof linkage)")
                
        if errors:
            with st.expander("Warnings (Skipped Angles)"):
                for e in errors:
                    st.warning(e)

elif mechanism == "Slider-Crank Mechanism":
    st.header("Slider-Crank Mechanism")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Link Lengths")
        r = st.number_input("Crank length r (m)", value=0.5, min_value=0.01)
        l = st.number_input("Connecting-rod l (m)", value=2.0, min_value=0.01)
        
        st.subheader("Kinematic Parameters")
        omega = st.number_input("Crank angular velocity ω (rad/s)", value=10.0)
        
        start_angle = st.number_input("Start Angle (°)", value=0)
        end_angle = st.number_input("End Angle (°)", value=360)
        step = st.number_input("Step (°)", value=5, min_value=1)
        
    with col2:
        thetas = np.arange(start_angle, end_angle + 1e-9, step)
        results = []
        errors = []
        
        for t in thetas:
            try:
                # Returns: phi_deg, omega2, alpha2, v_slider, a_slider
                phi, w2, a2, vs, a_s = slider_crank_kinematics(r, l, omega, t)
                results.append({
                    "θ (°)": t,
                    "φ (°)": phi,
                    "ω₂ (rad/s)": w2,
                    "α₂ (rad/s²)": a2,
                    "vₛ (m/s)": vs,
                    "aₛ (m/s²)": a_s
                })
            except Exception as e:
                errors.append(f"θ={t}°: {str(e)}")
                
        if len(results) > 0:
            df = pd.DataFrame(results)
            st.write("### Kinematic Data")
            st.dataframe(df.style.format("{:.3f}"), width='stretch')
            
            st.write("### Plots")
            
            tab1, tab2, tab3 = st.tabs(["Connecting Rod Angle & Speed", "Acceleration", "Slider Velocity & Acceleration"])
            
            with tab1:
                st.line_chart(df.set_index("θ (°)")[["φ (°)", "ω₂ (rad/s)"]])
            
            with tab2:
                st.line_chart(df.set_index("θ (°)")[["α₂ (rad/s²)"]])
                
            with tab3:
                st.line_chart(df.set_index("θ (°)")[["vₛ (m/s)", "aₛ (m/s²)"]])
                
        if errors:
            with st.expander("Geometry Warnings (Skipped Angles)"):
                for e in errors:
                    st.warning(e)

st.sidebar.markdown("---")
st.sidebar.info("Upload standard mechanisms script for dynamic visualization and data generation. Uses Streamlit for real-time adjustments.")
