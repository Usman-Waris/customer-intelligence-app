import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Advanced Behavioral Insights", layout="wide")

# 1. Load Model and Scaler
@st.cache_resource # Isse model baar baar load nahi hoga, app fast chalegi
def load_assets():
    try:
        model = joblib.load("customer_behavior_model_final.pkl")
        scaler = joblib.load("customer_scaler_final.pk1")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

if model is None:
    st.error("❌ Model ya Scaler file nahi mili. Pehle training script run karein.")
    st.stop()

st.title("🎯 Customer Behavior Intelligence")
st.markdown("According to specific features of dataset analyzed the customer segments:")

# 2. UI Layout for Inputs
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        order_freq = st.number_input("Order Frequency (Orders/Month)", min_value=0.0, format="%.2f")
        browsing = st.number_input("Browsing Intensity (Visits/Month)", min_value=0.0, format="%.2f")

    with col2:
        aov = st.number_input("Avg Order Value (AOV $)", min_value=0.0, format="%.2f")
        recency = st.number_input("Recency (Days since last order)", min_value=0, step=1)

    with col3:
        session_time = st.number_input("Avg Session Time (min)", min_value=0.0, format="%.2f")

st.divider()

# 3. Prediction Logic
if st.button("🚀 Analyze Customer Segment", use_container_width=True):
    # Input data preparation
    input_data = np.array([[order_freq, browsing, aov, recency, session_time]])
    scaled_data = scaler.transform(input_data)
    cluster = int(model.predict(scaled_data)[0])
    
    # Define Segment Labels & Actions
    cluster_info = {
        0: {"Name": "Churn Risk / Inactive", "Action": "Send a 'We Miss You' discount coupon immediately!", "Color": "error"},
        1: {"Name": "Loyal VIP", "Action": "Give early access to new products & loyalty points.", "Color": "success"},
        2: {"Name": "Active Browser / Frequent", "Action": "Show targeted ads to convert high browsing into sales.", "Color": "warning"},
        3: {"Name": "Patient Researcher", "Action": "Send reminder emails with product benefits.", "Color": "info"}
    }
    
    current_segment = cluster_info.get(cluster, {"Name": f"Cluster {cluster}", "Action": "Review manually", "Color": "info"})

    # --- Display Results ---
    st.subheader(f"📊 Prediction: {current_segment['Name']}")
    
    # Dynamic Alert Box based on Segment
    if cluster == 1:
        st.success(f"💎 **VIP Customer:** High spend and recently active. \n\n **Action:** {current_segment['Action']}")
    elif cluster == 0:
        st.error(f"⚠️ **At Risk:** High value but hasn't visited in a long time. \n\n **Action:** {current_segment['Action']}")
    elif cluster == 2:
        st.warning(f"⚡ **Frequent Visitor:** High engagement but quick sessions. \n\n **Action:** {current_segment['Action']}")
    elif cluster == 3:
        st.info(f"🧐 **Researcher:** Spends a lot of time browsing but orders less often. \n\n **Action:** {current_segment['Action']}")

    st.write("### 📉 Comparative Insights (vs. Market Average)")
    
    # Comparison Metrics with your reported averages
    c1, c2, c3 = st.columns(3)
    
    # Recency comparison (Lower is better, so delta_color="inverse")
    avg_recency = 222 
    c1.metric("Recency", f"{recency} Days", delta=f"{int(recency - avg_recency)} vs Avg", delta_color="inverse")
    
    # AOV comparison
    avg_aov = 977
    c2.metric("AOV", f"${aov}", delta=f"${round(aov - avg_aov, 2)} vs Avg")
    
    # Session Time comparison
    avg_session = 14.2
    c3.metric("Session Time", f"{session_time} min", delta=f"{round(session_time - avg_session, 2)} vs Avg")