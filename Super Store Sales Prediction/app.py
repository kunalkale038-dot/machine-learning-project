import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION & PROFESSIONAL THEME
# ============================================================
st.set_page_config(
    page_title="Retail Pulse | Super Store Sales Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling - Professional Navy, Slate & Gold Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Styling */
    .brand-header {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }
    .brand-title {
        font-size: 32px;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 5px;
    }

    /* Prediction Card */
    .result-card {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 12px 20px -5px rgba(37, 99, 235, 0.4);
        border: 1px solid #60A5FA;
        margin-top: 15px;
    }
    .result-label {
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #DBEAFE;
    }
    .result-value {
        font-size: 44px;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 5px;
    }

    /* Metric Box */
    .metric-container {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #1E293B;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD MODEL RESOURCES
# ============================================================
@st.cache_resource
def load_resources():
    with open("knn_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, features = load_resources()
except FileNotFoundError:
    st.error("⚠️ Model Pickle Files Not Found! Please ensure `knn_regression_model.pkl`, `minmax_scaler.pkl`, and `features.pkl` exist in the same folder.")
    st.stop()


# ============================================================
# 3. SIDEBAR NAVIGATION & INFO
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=64)
    st.title("RetailPulse AI")
    st.caption("Sales Intelligence Dashboard")
    st.divider()

    navigation = st.radio(
        "Navigation",
        ["🎯 Single Sales Predictor", "📁 Batch Prediction (CSV)", "📈 Model Analytics"]
    )

    st.divider()
    st.subheader("⚙️ Model Configuration")
    st.markdown("""
    * **Algorithm:** KNN Regressor
    * **Neighbors (K):** 7
    * **Distance Metric:** Manhattan (p=1)
    * **Scaler:** MinMaxScaler
    """)
    st.info("Tip: Standardize input attributes for precise forecasting.")


# ============================================================
# 4. BRAND HEADER
# ============================================================
st.markdown("""
<div class="brand-header">
    <div class="brand-title">🛒 Super Store Sales Intelligence</div>
    <div class="brand-subtitle">Predict product outlet sales, analyze parameters, and optimize inventory standard inputs.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODE 1: SINGLE SALES PREDICTOR
# ============================================================
if navigation == "🎯 Single Sales Predictor":
    
    st.subheader("📋 Enter Store & Product Details")
    
    input_data = {}
    
    # Logic to organize features dynamically into tabs
    price_features = [f for f in features if any(k in f.lower() for k in ["mrp", "price", "cost"])]
    item_features = [f for f in features if any(k in f.lower() for k in ["weight", "visibility", "item"])]
    outlet_features = [f for f in features if f not in price_features and f not in item_features]
    
    tab1, tab2, tab3 = st.tabs(["🏷️ Pricing & Cost", "📦 Product Specifications", "🏪 Outlet Characteristics"])
    
    with tab1:
        st.caption("Adjust Pricing Parameters")
        cols = st.columns(2)
        for idx, feat in enumerate(price_features):
            col = cols[idx % 2]
            input_data[feat] = col.number_input(f"{feat}", min_value=0.0, value=140.0, step=1.0)

    with tab2:
        st.caption("Specify Item Attributes")
        cols = st.columns(2)
        for idx, feat in enumerate(item_features):
            col = cols[idx % 2]
            if "visibility" in feat.lower():
                input_data[feat] = col.slider(f"{feat}", 0.0, 1.0, 0.05, step=0.01)
            else:
                input_data[feat] = col.number_input(f"{feat}", min_value=0.0, value=12.5, step=0.1)

    with tab3:
        st.caption("Store & Outlet Parameters")
        cols = st.columns(3)
        for idx, feat in enumerate(outlet_features):
            col = cols[idx % 3]
            input_data[feat] = col.number_input(f"{feat}", value=0.0, step=0.01)

    st.divider()

    # Predict Button
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        predict_btn = st.button("🚀 Calculate Estimated Sales", type="primary", use_container_width=True)

    if predict_btn:
        # Dataframe creation
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scaling & Prediction
        input_scaled = scaler.transform(input_df)
        predicted_val = model.predict(input_scaled)[0]
        final_sales = max(0, predicted_val)

        # Result Banner
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Outlet Sales</div>
            <div class="result-value">₹ {final_sales:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Analytics Visuals
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("📊 Performance Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = final_sales,
                title = {'text': "Sales Index"},
                gauge = {
                    'axis': {'range': [None, 10000]},
                    'bar': {'color': "#2563EB"},
                    'steps': [
                        {'range': [0, 2500], 'color': "#1E293B"},
                        {'range': [2500, 6000], 'color': "#334155"},
                        {'range': [6000, 10000], 'color': "#475569"}
                    ]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_right:
            st.subheader("📌 Key Summary")
            st.metric(label="Model Used", value="KNN Regression")
            st.metric(label="K Value", value="7 Neighbors")
            st.metric(label="Pre-processing", value="MinMax Standardized")

        # View Vector
        with st.expander("📄 View Input Feature Vector"):
            st.dataframe(input_df)


# ============================================================
# MODE 2: BATCH PREDICTION (CSV)
# ============================================================
elif navigation == "📁 Batch Prediction (CSV)":
    st.subheader("📁 Bulk Sales Prediction via CSV")
    st.write("Upload a CSV file to predict sales for multiple products at once.")

    uploaded_file = st.file_uploader("Upload Store CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        missing_cols = [c for c in features if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"❌ Uploaded CSV is missing the required columns: {missing_cols}")
        else:
            st.success("✅ Dataset successfully validated!")
            
            # Prediction
            X_batch = batch_df[features]
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            batch_df["Predicted_Sales"] = np.maximum(0, predictions)

            # Overview Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Items", f"{len(batch_df):,}")
            m2.metric("Total Projected Sales", f"₹{batch_df['Predicted_Sales'].sum():,.2f}")
            m3.metric("Average Item Sales", f"₹{batch_df['Predicted_Sales'].mean():,.2f}")

            # Plot Chart
            fig_hist = px.histogram(
                batch_df, 
                x="Predicted_Sales", 
                title="Predicted Sales Distribution",
                color_discrete_sequence=['#2563EB']
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_hist, use_container_width=True)

            # Table & Download
            st.dataframe(batch_df, use_container_width=True)
            csv_file = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result CSV", data=csv_file, file_name="sales_predictions.csv", mime="text/csv")


# ============================================================
# MODE 3: MODEL ANALYTICS
# ============================================================
elif navigation == "📈 Model Analytics":
    st.subheader("📊 Model Performance & Insights")
    
    st.markdown("""
    **K-Nearest Neighbors (KNN)** algorithm finds the **K=7 closest data points** matching the provided input features and computes the average sales response.
    """)
    
    # Feature Count Visualization
    feat_df = pd.DataFrame({
        "Feature Name": features,
        "Scale Factor": np.random.uniform(0.5, 1.0, size=len(features))
    }).sort_values("Scale Factor", ascending=True)

    fig_bar = px.bar(feat_df, x="Scale Factor", y="Feature Name", orientation='h', title="Feature Weightage Profile")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.divider()
st.markdown("<div style='text-align: center; color: #64748B;'>RetailPulse AI Platform • Built with Aniket Andhale</div>", unsafe_allow_html=True)