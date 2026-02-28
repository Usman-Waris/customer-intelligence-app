import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Behavioral Insights", layout="wide")

# 1. Load Model and Scaler
@st.cache_resource
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
st.markdown("Customer ko un ki behavioral features ke hisab se segments mein classify karte hain:")

# 2. UI Layout for Inputs
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        order_freq = st.number_input("Order Frequency (Orders/Month)", min_value=0.0, format="%.2f")
        browsing = st.number_input("Browsing Intensity (Visits/Month)", min_value=0.0, format="%.2f")

    with col2:
        aov = st.number_input("Avg Order Quarter Value (AOV $)", min_value=0.0, format="%.2f")
        recency = st.number_input("Recency (Days since last order)", min_value=0, step=1)

    with col3:
        session_time = st.number_input("Avg Session Time (min)", min_value=0.0, format="%.2f")

st.divider()

# 3. Prediction Logic
if st.button("🚀 Analyze Customer Segment", use_container_width=True):

    # Ensure expected number of features
    expected_features = scaler.n_features_in_
    all_inputs = [order_freq, browsing, aov, recency, session_time]
    input_data = np.array([all_inputs[:expected_features]])

    # Transform and predict
    scaled_data = scaler.transform(input_data)
    raw_cluster = int(model.predict(scaled_data)[0])

    # --- Logic to assign semantic labels to clusters ---
    centers = scaler.inverse_transform(model.cluster_centers_)
    centers_df = pd.DataFrame(
        centers,
        columns=["Order_Frequency", "Browsing", "AOV", "Recency", "Session_Time"][:expected_features]
    )

    # Threshold heuristic: if all inputs are near zero, treat as inactive
    if all(x <= 1e-6 for x in [order_freq, browsing, aov, recency, session_time]):
        cluster = -1  # special code for "near‑zero / no activity"
    else:
        cluster = raw_cluster

    # VIP = High AOV + Low Recency
    if "AOV" in centers_df.columns and "Recency" in centers_df.columns:
        vip_cluster = centers_df.sort_values(
            by=["AOV", "Recency"],
            ascending=[False, True]
        ).index[0]
    else:
        vip_cluster = 0

    # Churn = High Recency (inactive)
    churn_col = "Recency" if "Recency" in centers_df.columns else centers_df.columns[-1]
    churn_cluster = centers_df.sort_values(by=churn_col, ascending=False).index[0]

    # Active Browser = High Browsing
    if "Browsing" in centers_df.columns:
        browser_cluster = centers_df["Browsing"].idxmax()
    else:
        browser_cluster = vip_cluster

    # Remaining cluster (researcher)
    remaining = list(set(range(len(centers_df))) - {vip_cluster, churn_cluster, browser_cluster})
    researcher_cluster = remaining[0] if remaining else vip_cluster

    cluster_info = {
        vip_cluster: {
            "Name": "Loyal VIP",
            "Action": "Give early access to new products & loyalty points.",
            "Color": "success"
        },
        churn_cluster: {
            "Name": "Churn Risk / Inactive",
            "Action": "Send a 'We Miss You' discount coupon immediately!",
            "Color": "error"
        },
        browser_cluster: {
            "Name": "Active Browser / Frequent",
            "Action": "Show targeted ads to convert high browsing into sales.",
            "Color": "warning"
        },
        researcher_cluster: {
            "Name": "Patient Researcher",
            "Action": "Send reminder emails with product benefits.",
            "Color": "info"
        }
    }

    # Force near‑zero input to be "Churn Risk / Inactive"
    if cluster == -1:
        final_segment = {
            "Name": "New / No Activity",
            "Action": "Treat as inactive or new user; start with engagement campaigns.",
            "Color": "error"
        }
    else:
        final_segment = cluster_info.get(
            cluster,
            {
                "Name": f"Cluster {cluster}",
                "Action": "Review manually",
                "Color": "info"
            }
        )

    # --- Display Results ---
    st.subheader(f"📊 Prediction: {final_segment['Name']}")

    if final_segment["Color"] == "success":
        st.success(f"💎 **VIP Customer**\n\nAction: {final_segment['Action']}")
    elif final_segment["Color"] == "error":
        st.error(f"⚠️ **At Risk / No‑Activity Customer**\n\nAction: {final_segment['Action']}")
    elif final_segment["Color"] == "warning":
        st.warning(f"⚡ **Frequent Visitor**\n\nAction: {final_segment['Action']}")
    else:
        st.info(f"🧐 **Researcher Type**\n\nAction: {final_segment['Action']}")

    # --- Comparative Insights ---
    st.write("### 📉 Comparative Insights (vs. Market Average)")

    c1, c2, c3 = st.columns(3)

    avg_recency = 222
    c1.metric("Recency", f"{recency} Days",
              delta=f"{int(recency - avg_recency)} vs Avg",
              delta_color="inverse")

    avg_aov = 977
    c2.metric("AOV", f"${aov}",
              delta=f"${round(aov - avg_aov, 2)} vs Avg")

    avg_session = 14.2
    c3.metric("Session Time", f"{session_time} min",
              delta=f"{round(session_time - avg_session, 2)} vs Avg")

    # --- Simple feature‑importance style bar (cluster centers) ---
    st.write("### 🧩 Cluster Centers Overview")

    # Keep only the first 5 features (for plotting)
    plot_cols = centers_df.columns[:5]
    fig, ax = plt.subplots(figsize=(8, 3))
    centers_df[plot_cols].plot(kind="bar", ax=ax, legend=False, alpha=0.8)
    ax.set_title("Cluster Centers (per feature)")
    ax.set_ylabel("Value")
    ax.set_xlabel("Cluster")
    st.pyplot(fig)
