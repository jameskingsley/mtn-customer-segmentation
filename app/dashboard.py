import streamlit as st
import requests

# Set page layout to wide for a professional corporate feel
st.set_page_config(
    page_title="MTN Customer Segmentation Portal",
    page_icon="",
    layout="wide"
)

# Main Title and Header
st.title("MTN Customer Segmentation Portal")
st.markdown("### Real-Time Subscriber Persona Classification Engine")
st.markdown("---")

# Backend API endpoint configuration
API_URL = "https://mtn-customer-segmentation.onrender.com/predict/segment"

# Input Sidebar (left) vs Visualization Display (right)
col_input, col_display = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("Input Subscriber Metrics")
    st.info("Enter current usage attributes to classify the user segment in real time.")
    
    # Form-based entry to prevent layout flashing on every user keystroke
    with st.form("subscriber_metrics_form"):
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)
        tenure = st.number_input("Customer Tenure (Months)", min_value=0, max_value=240, value=24, step=1)
        satisfaction = st.slider("Satisfaction Rate", min_value=1, max_value=5, value=4, step=1)
        unit_price = st.number_input("Unit Price (₦)", min_value=0, value=5000, step=500)
        data_usage = st.number_input("Data Usage (GB)", min_value=0.0, value=25.0, step=5.0)
        total_revenue = st.number_input("Total Revenue (₦)", min_value=0.0, value=120000.0, step=5000.0)
        
        device_label = st.selectbox(
            "Primary MTN Device Type",
            ["Mobile SIM Card", "Broadband MiFi", "4G Router", "5G Broadband Router"]
        )
        
        # Map human-readable device back to production integer tier sequence
        device_tier_map = {
            "Mobile SIM Card": 1,
            "Broadband MiFi": 2,
            "4G Router": 3,
            "5G Broadband Router": 4
        }
        device_tier = device_tier_map[device_label]
        
        submit_btn = st.form_submit_button("Run Segment Analysis", use_container_width=True)

with col_display:
    st.subheader("Persona Classification Summary")
    
    if submit_btn:
        # Build payload corresponding to the CustomerInput Pydantic model
        payload = {
            "Age": float(age),
            "Customer_Tenure_Months": float(tenure),
            "Satisfaction_Rate": float(satisfaction),
            "Unit_Price": float(unit_price),
            "Data_Usage_GB": float(data_usage),
            "Total_Revenue": float(total_revenue),
            "Device_Tier": int(device_tier)
        }
        
        try:
            with st.spinner("Querying ClearML backend model engine..."):
                response = requests.post(API_URL, json=payload, timeout=30)
                
            if response.status_code == 200:
                result = response.json()
                cluster_id = result.get("predicted_cluster_id")
                persona_name = result.get("assigned_persona")
                
                # Contextual coloring configuration for UI presentation cards
                color_map = {
                    0: {"bg": "#E8F4F8", "text": "#1A5276", "desc": "High volume, base-tier consumers utilizing mobile SIM lines on standard subscriptions."},
                    1: {"bg": "#FCF3CF", "text": "#7E5109", "desc": "Mid-tier router users with growing loyalty profiles but shorter active tenures."},
                    2: {"bg": "#E8F8F5", "text": "#0E6251", "desc": "Premium high-revenue enterprise lines driving massive data usage volumes over 5G hardware."},
                    3: {"bg": "#F5EEF8", "text": "#4A235A", "desc": "Power data hubs operating on unlimited monthly plans across standard 4G installations."}
                }
                
                cfg = color_map.get(cluster_id, {"bg": "#F2F4F4", "text": "#2C3E50", "desc": "Standard Account Segment"})
                
                # Beautiful raw metric layout display
                st.metric(label="Assigned Cluster ID", value=f"Cluster {cluster_id}")
                
                # Inject a stylized HTML persona container box
                st.markdown(
                    f"""
                    <div style="background-color: {cfg['bg']}; padding: 25px; border-radius: 10px; border-left: 8px solid {cfg['text']};">
                        <h2 style="color: {cfg['text']}; margin-top: 0;">{persona_name}</h2>
                        <p style="font-size: 16px; color: #333333; line-height: 1.5;">{cfg['desc']}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Render clean summary telemetry cards of current entries
                st.markdown("### Telemetry Breakdown")
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("ARPU Implied", f"₦{total_revenue:,.2f}")
                metrics_cols[1].metric("Bandwidth Footprint", f"{data_usage} GB")
                metrics_cols[2].metric("Hardware Class", device_label)
                
            else:
                st.error(f"Backend API error code: {response.status_code}. Content: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Connection Refused! Make sure your FastAPI backend engine is actively running at http://127.0.0.1:8000 via terminal.")
    else:
        st.warning("Adjust parameters and press 'Run Segment Analysis' to kick off the prediction pipeline.")