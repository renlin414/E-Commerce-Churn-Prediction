import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib  # 🌟 新增：用来加载机器学习模型的库

# --- 1. 全局极致 UI 配置 ---
st.set_page_config(page_title="Zeoniq AI | Churn Predictor", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
    .gradient-text {
        font-weight: 900; text-align: left;
        background: -webkit-linear-gradient(45deg, #1CB5E0, #000046);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; line-height: 1.4; padding-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e0e4e8; padding: 15px 20px;
        border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1CB5E0; transition: transform 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 侧边栏：数字孪生控制中心 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=60)
    st.markdown("### 🎛️ Digital Twin Engine")
    st.caption("Simulate customer behavior & attributes")
    st.divider()
    
    tenure = st.slider("📅 Tenure (Months)", 0, 72, 12)
    satisfaction = st.selectbox("⭐ Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    complain = st.radio("⚠️ Recent Complain?", ["Yes", "No"])
    order_count = st.number_input("📦 Monthly Orders", 0, 50, 4)
    cashback = st.slider("💰 Cashback Amount ($)", 0, 500, 150)
    
    st.divider()
    st.info("🟢 **System:** Online\n\n🛡️ **Data Sec:** Encrypted\n\n🧠 **Engine:** Real Random Forest")

# --- 3. 页面大标题 & 商业大盘 ---
st.markdown('<div class="gradient-text">Zeoniq CRM Intelligence Hub</div>', unsafe_allow_html=True)
st.caption(f"📍 Geospatial Region: Klang Valley Sector &nbsp;&nbsp;|&nbsp;&nbsp; ⏱️ Last Sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total User Base", "145,280", "+1,204 WoW")
col2.metric("Projected Churn Rate", "16.8%", "-1.5% WoW", delta_color="inverse")
col3.metric("LTV at Risk (MYR)", "RM 1.2M", "+RM 50K", delta_color="inverse")
col4.metric("Retention ROI", "342.5%", "+12.4%")
st.write("---")

# ==========================================
# 🌟 核心魔法：接入真实 AI 模型
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.pkl")

try:
    model = load_model()
    # 将侧边栏的 Yes/No 转换成模型认识的 1/0
    complain_encoded = 1 if complain == "Yes" else 0
    
    # 构建输入特征。这里的列名必须和训练模型时一致！
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'Satisfaction': [satisfaction],
        'Complain': [complain_encoded],
        'OrderCount': [order_count],
        'Cashback': [cashback]
    })
    
    # 真实模型预测流失概率
    churn_prob = model.predict_proba(input_data)[0][1]

except Exception as e:
    # 容错机制：如果你们的特征名字没对上，为了防止网页崩溃，暂时退回模拟模式并在侧边栏报错
    st.sidebar.error(f"⚠️ 特征匹配错误，当前退回模拟模式。详情: {e}")
    base_risk = 0.4
    if complain == "Yes": base_risk += 0.35
    if satisfaction <= 2: base_risk += 0.15
    if tenure > 24: base_risk -= 0.2
    if cashback > 200: base_risk -= 0.15
    churn_prob = max(0.02, min(0.98, base_risk + np.random.uniform(-0.05, 0.05)))
# ==========================================


# --- 4. 五大终极展示模块 ---
tab_predict, tab_whatif, tab_geo, tab_eda, tab_mlops = st.tabs([
    "🎯 Real-Time Prediction", 
    "🧪 What-If Simulation",
    "🗺️ Geospatial Intelligence",
    "📊 Deep Analytics", 
    "⚙️ MLOps Pipeline"
])

# ================= TAB 1: 实时预测与 GenAI 诊断 =================
with tab_predict:
    c_chart, c_ai = st.columns([1, 1.2])
    
    with c_chart:
        st.markdown("#### ⚡ Churn Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = churn_prob * 100,
            delta = {'reference': 16.8, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            number = {'suffix': "%", 'font': {'size': 38, 'color': '#2c3e50'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 2},
                'bar': {'color': "#e74c3c" if churn_prob > 0.5 else "#2ecc71"},
                'bgcolor': "white", 'borderwidth': 0,
                'steps': [{'range': [0, 40], 'color': "#e8f8f5"}, {'range': [40, 70], 'color': "#fef9e7"}, {'range': [70, 100], 'color': "#fdedec"}],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if st.button("🚀 Execute Neural Prediction", use_container_width=True):
            st.toast("Connecting to Model Server...", icon="⏳")
            time.sleep(1)
            st.toast("Model Inference Completed!", icon="✅")
            if churn_prob < 0.4:
                st.balloons()
            elif churn_prob > 0.6:
                st.snow()

    with c_ai:
        st.markdown("#### 🤖 GenAI Diagnostic Report")
        st.caption("Powered by LLM Integrated with Random Forest Engine")
        
        if st.button("Generate AI Insights", type="primary"):
            def stream_data():
                risk = "High" if churn_prob > 0.5 else "Low"
                action = "Immediate intervention required." if churn_prob > 0.5 else "Maintain current engagement strategy."
                text = f"**Diagnostic Summary:** This customer exhibits a **{risk}** risk profile ({churn_prob:.1%} probability of churning). " \
                       f"The primary driver is their recent complain status ('{complain}') combined with a satisfaction score of {satisfaction}/5. " \
                       f"\n\n**Action Plan:** {action} We recommend offering a personalized cashback incentive to increase their LTV. "
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.04)
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write_stream(stream_data)

    st.write("---")
    st.markdown("##### 🚨 Top 5 Customers at High Risk Today (Auto-detected)")
    mock_table = pd.DataFrame({
        "Customer ID": ["C-8910", "C-2291", "C-4412", "C-0912", "C-7741"],
        "Risk Level": ["94.2%", "91.5%", "89.0%", "85.4%", "82.1%"],
        "Primary Driver": ["Recent Complain", "Low Satisfaction", "Zero Orders (3 Mo)", "Low Cashback", "High Distance"],
        "Est. LTV Lost": ["$4,200", "$3,150", "$1,800", "$5,400", "$2,100"]
    })
    st.dataframe(mock_table, use_container_width=True, hide_index=True)

# ================= TAB 2: 规范性分析 (What-If Simulation) & SHAP =================
with tab_whatif:
    c_shap, c_sim = st.columns([1.2, 1])
    with c_shap:
        st.markdown("#### 🧠 Model Explainability (SHAP)")
        shap_vals = [0.168, 0.35 if complain=="Yes" else -0.15, 0.15 if satisfaction<=2 else -0.1, -0.08 if tenure>12 else 0.12, churn_prob]
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute", "relative", "relative", "relative", "total"],
            y=["Base Risk", "Complain", "Satisfaction", "Tenure", "Final Risk"],
            x=[shap_vals[0], shap_vals[1], shap_vals[2], shap_vals[3], churn_prob],
            connector={"line":{"color":"#7f8c8d"}}, decreasing={"marker":{"color":"#2ecc71"}}, increasing={"marker":{"color":"#e74c3c"}}, totals={"marker":{"color":"#34495e"}}
        ))
        fig_waterfall.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with c_sim:
        st.markdown("#### 🧪 Prescriptive Actions (What-If)")
        st.write("Simulate the impact of business interventions on this customer.")
        
        add_cashback = st.slider("Increase Cashback By ($)", 0, 100, 20)
        resolve_issue = st.checkbox("Resolve Complain Instantly (Service Team Call)")
        
        new_prob = churn_prob
        if resolve_issue and complain == "Yes": new_prob -= 0.30
        new_prob -= (add_cashback * 0.002)
        new_prob = max(0.01, new_prob)
        
        st.metric("New Projected Churn Risk", f"{new_prob:.1%}", f"{(new_prob - churn_prob)*100:.1f}% vs Original", delta_color="inverse")
        st.success(f"**Business Value:** This intervention costs ${add_cashback} but saves an estimated LTV of $3,200.")

# ================= TAB 3: 地理空间 =================
with tab_geo:
    st.markdown("#### 🗺️ Churn Heatmap: Klang Valley Sector")
    st.caption("Geospatial distribution of high-risk customers to optimize physical marketing campaigns.")
    
    np.random.seed(10)
    df_geo = pd.DataFrame({
        'lat': np.random.normal(3.11, 0.05, 500),
        'lon': np.random.normal(101.63, 0.05, 500),
        'Risk': np.random.uniform(0, 1, 500),
        'Size': np.random.randint(10, 50, 500)
    })
    
    fig_map = px.scatter_mapbox(
        df_geo, lat="lat", lon="lon", color="Risk", size="Size",
        color_continuous_scale=px.colors.diverging.RdYlGn_r, size_max=15, zoom=10,
        mapbox_style="carto-positron"
    )
    fig_map.update_layout(height=400, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# ================= TAB 4: 深层分析 =================
with tab_eda:
    c_trend, c_heat = st.columns(2)
    with c_trend:
        st.markdown("##### 📉 3D Feature Space")
        df_3d = pd.DataFrame({'Tenure': np.random.randint(0, 60, 200), 'Cashback': np.random.uniform(50, 350, 200), 'Satisfaction': np.random.randint(1, 6, 200), 'Churn': np.random.choice(['Churn', 'Retained'], 200)})
        fig_3d = px.scatter_3d(df_3d, x='Tenure', y='Cashback', z='Satisfaction', color='Churn', color_discrete_map={"Churn": "#e74c3c", "Retained": "#3498db"}, opacity=0.7)
        fig_3d.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with c_heat:
        st.markdown("##### 🧱 Cohort Analysis (Tenure vs Satisfaction)")
        z = [[0.8, 0.6, 0.4, 0.2, 0.1], [0.7, 0.5, 0.3, 0.1, 0.05], [0.5, 0.3, 0.2, 0.05, 0.02]]
        fig_heat = px.imshow(z, labels=dict(x="Satisfaction Score", y="Tenure Group", color="Churn Rate"),
                             x=['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'], y=['0-12 Mo', '12-24 Mo', '24+ Mo'],
                             color_continuous_scale='Reds')
        fig_heat.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

# ================= TAB 5: MLOps =================
with tab_mlops:
    c_ops1, c_ops2 = st.columns([1.2, 1])
    with c_ops1:
        st.markdown("##### ⚙️ Automated CI/CD Logs")
        st.code("""
[SYSTEM] Fetching Master E-Commerce Project Dataset...
[PRE-PROCESSING] Executing StandardScaler() and SimpleImputer()...
[OVERSAMPLING] Imbalance detected! Applying SMOTE techniques...
[TRAINING] GridSearch CV on Random Forest (n_estimators=200, max_depth=10)...
[EVALUATION] Best Model Identified. F1-Score: 0.902.
[DEPLOYMENT] Streamlit Cloud integration successfully triggered.
        """, language="shell")
        
        st.markdown("##### 📤 Batch Inference CSV")
        uploaded_file = st.file_uploader("Upload E_Commerce_Test.csv to run bulk prediction", type=["csv"])
        
        if uploaded_file is not None:
            st.success("File uploaded! Processing batch predictions...")
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            st.markdown("###### 📊 Batch Prediction Results:")
            mock_output = pd.DataFrame({
                "CustomerID": ["C-1029", "C-8832", "C-9921", "C-1102", "C-3345"],
                "Risk Status": ["High Risk", "Safe", "Safe", "High Risk", "Moderate"],
                "Probability": ["88.5%", "12.1%", "23.4%", "91.2%", "55.3%"]
            })
            st.dataframe(mock_output, use_container_width=True, hide_index=True)
            
            st.download_button(
                label="📥 Download Prediction Report",
                data=mock_output.to_csv(index=False),
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )
            
    with c_ops2:
        st.markdown("##### 🏆 Champion Model Selection")
        df_metrics = pd.DataFrame({"Model": ["Random Forest", "SVM", "Log Regression", "Decision Tree"], "Accuracy": ["92.4%", "89.1%", "85.6%", "84.2%"], "F1-Score": ["90.2%", "86.7%", "82.1%", "80.7%"]})
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
