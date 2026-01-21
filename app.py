import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from sklearn.metrics import confusion_matrix

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model_and_data():
    try:
        # ABSOLUTE path to your project
        base_path = '/Users/kashishpatel/Desktop/customer-churn-project'
        
        # Load model
        with open(f'{base_path}/models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(f'{base_path}/models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load test data
        X_test = pd.read_csv(f'{base_path}/data/processed/X_test.csv')
        y_test = pd.read_csv(f'{base_path}/data/processed/y_test.csv').values.ravel()
        
        # Load scaler
        with open(f'{base_path}/data/processed/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, metadata, X_test, y_test, scaler
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load everything
model, metadata, X_test, y_test, scaler = load_model_and_data()

# Calculate predictions
@st.cache_data
def get_predictions():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return y_pred, y_pred_proba

y_pred, y_pred_proba = get_predictions()

# Sidebar
st.sidebar.markdown("# 🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard Overview", "🔍 Customer Analysis", "💡 Model Insights", "📈 Business Impact"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Info")
st.sidebar.info(f"""
**Model**: {metadata['model_name']}  
**ROC AUC**: {metadata['metrics']['roc_auc']:.3f}  
**F1 Score**: {metadata['metrics']['f1']:.3f}  
**Precision**: {metadata['metrics']['precision']:.3f}  
**Recall**: {metadata['metrics']['recall']:.3f}
""")

# ========== PAGE 1: DASHBOARD OVERVIEW ==========
if page == "📊 Dashboard Overview":
    st.markdown('<p class="main-header">📊 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(X_test)
    high_risk = (y_pred_proba > 0.7).sum()
    medium_risk = ((y_pred_proba > 0.4) & (y_pred_proba <= 0.7)).sum()
    low_risk = (y_pred_proba <= 0.4).sum()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("High Risk (>70%)", f"{high_risk:,}", delta=f"{high_risk/total_customers*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Medium Risk (40-70%)", f"{medium_risk:,}", delta=f"{medium_risk/total_customers*100:.1f}%")
    with col4:
        st.metric("Low Risk (<40%)", f"{low_risk:,}", delta=f"{low_risk/total_customers*100:.1f}%", delta_color="normal")
    
    st.markdown("---")
    
    # Two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        
        # Risk distribution pie chart
        risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_counts = [low_risk, medium_risk, high_risk]
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_categories,
            values=risk_counts,
            hole=0.4,
            marker=dict(colors=['#2ca02c', '#ff7f0e', '#d62728'])
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Churn Probability Distribution")
        
        # Histogram of churn probabilities
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_pred_proba,
            nbinsx=50,
            marker=dict(color='steelblue', line=dict(color='white', width=1))
        ))
        fig.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=400,
            showlegend=False
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top high-risk customers table
    st.subheader("🚨 Top 20 High-Risk Customers Requiring Immediate Attention")
    
    high_risk_df = pd.DataFrame({
        'Customer_Index': X_test.index,
        'Churn_Probability': y_pred_proba,
        'Predicted_Churn': ['Yes' if p == 1 else 'No' for p in y_pred],
        'Actual_Churn': ['Yes' if p == 1 else 'No' for p in y_test]
    }).sort_values('Churn_Probability', ascending=False).head(20)
    
    # Color code the probability
    def color_probability(val):
        if val > 0.7:
            return 'background-color: #ffcccc'
        elif val > 0.4:
            return 'background-color: #ffe6cc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = high_risk_df.style.applymap(
        color_probability, 
        subset=['Churn_Probability']
    ).format({'Churn_Probability': '{:.1%}'})
    
    st.dataframe(styled_df, use_container_width=True, height=400)

# ========== PAGE 2: CUSTOMER ANALYSIS ==========
elif page == "🔍 Customer Analysis":
    st.markdown('<p class="main-header">🔍 Individual Customer Analysis</p>', unsafe_allow_html=True)
    
    # Customer selector
    st.subheader("Select a Customer to Analyze")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selection_method = st.radio(
            "Selection Method",
            ["By Risk Level", "By Customer Index"]
        )
    
    with col2:
        if selection_method == "By Risk Level":
            risk_level = st.selectbox(
                "Risk Level",
                ["Highest Risk", "High Risk (Random)", "Medium Risk (Random)", "Low Risk (Random)", "Lowest Risk"]
            )
            
            if risk_level == "Highest Risk":
                customer_idx = y_pred_proba.argmax()
            elif risk_level == "Lowest Risk":
                customer_idx = y_pred_proba.argmin()
            elif risk_level == "High Risk (Random)":
                high_risk_indices = np.where(y_pred_proba > 0.7)[0]
                customer_idx = np.random.choice(high_risk_indices) if len(high_risk_indices) > 0 else 0
            elif risk_level == "Medium Risk (Random)":
                medium_risk_indices = np.where((y_pred_proba > 0.4) & (y_pred_proba <= 0.7))[0]
                customer_idx = np.random.choice(medium_risk_indices) if len(medium_risk_indices) > 0 else 0
            else:
                low_risk_indices = np.where(y_pred_proba <= 0.4)[0]
                customer_idx = np.random.choice(low_risk_indices) if len(low_risk_indices) > 0 else 0
        else:
            customer_idx = st.number_input(
                "Customer Index",
                min_value=0,
                max_value=len(X_test)-1,
                value=0
            )
    
    # Get customer data
    customer_data = X_test.iloc[customer_idx]
    customer_prob = y_pred_proba[customer_idx]
    customer_pred = y_pred[customer_idx]
    customer_actual = y_test[customer_idx]
    
    st.markdown("---")
    
    # Customer info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customer Index", f"{X_test.index[customer_idx]}")
    with col2:
        risk_class = "High" if customer_prob > 0.7 else "Medium" if customer_prob > 0.4 else "Low"
        risk_color = "high-risk" if customer_prob > 0.7 else "medium-risk" if customer_prob > 0.4 else "low-risk"
        st.markdown(f"**Churn Probability**")
        st.markdown(f'<p class="{risk_color}">{customer_prob:.1%}</p>', unsafe_allow_html=True)
    with col3:
        st.metric("Predicted", "Will Churn" if customer_pred == 1 else "Will Stay")
    with col4:
        st.metric("Actual", "Churned" if customer_actual == 1 else "Stayed")
    
    st.markdown("---")
    
    # Feature values
    st.subheader("📋 Customer Feature Profile")
    
    # Show top features
    feature_df = pd.DataFrame({
        'Feature': customer_data.index,
        'Value': customer_data.values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(feature_df.head(len(feature_df)//2), use_container_width=True, height=400)
    
    with col2:
        st.dataframe(feature_df.tail(len(feature_df) - len(feature_df)//2), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Recommended Actions to Prevent Churn")
    
    if customer_prob > 0.7:
        st.error("⚠️ **HIGH RISK CUSTOMER** - Immediate intervention required!")
        
        recommendations = [
            "🎯 **Priority Action**: Contact customer within 24 hours",
            "💰 Offer loyalty discount (10-15% off monthly charges)",
            "📞 Schedule retention call with senior representative",
            "🎁 Provide upgrade to annual contract with benefits",
            "🛡️ Offer additional services (security, backup) at discounted rate",
            "📊 Review service quality and address any complaints"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    elif customer_prob > 0.4:
        st.warning("⚠️ **MEDIUM RISK CUSTOMER** - Proactive engagement recommended")
        
        recommendations = [
            "📧 Send personalized email with service tips",
            "🎁 Offer service bundle discount",
            "📞 Follow-up call within 1 week",
            "💬 Request feedback survey",
            "🌟 Highlight unused features they might benefit from"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    else:
        st.success("✅ **LOW RISK CUSTOMER** - Maintain engagement")
        
        recommendations = [
            "🎉 Send appreciation message",
            "⭐ Offer referral rewards program",
            "📬 Monthly newsletter with tips and updates",
            "💝 Loyalty rewards for continued service"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

# ========== PAGE 3: MODEL INSIGHTS ==========
elif page == "💡 Model Insights":
    st.markdown('<p class="main-header">💡 Model Insights & Explainability</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Feature Importance")
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importance['Importance'],
            y=feature_importance['Feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        fig.update_layout(
            title="Top 20 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model performance
    st.subheader("📈 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROC AUC Score", f"{metadata['metrics']['roc_auc']:.3f}")
        st.metric("Accuracy", f"{metadata['metrics']['accuracy']:.3f}")
    
    with col2:
        st.metric("Precision", f"{metadata['metrics']['precision']:.3f}")
        st.metric("Recall", f"{metadata['metrics']['recall']:.3f}")
    
    with col3:
        st.metric("F1 Score", f"{metadata['metrics']['f1']:.3f}")
    
    st.markdown("---")
    
    # Confusion matrix
    st.subheader("🎯 Prediction Accuracy")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Churn', 'Predicted: Churn'],
        y=['Actual: No Churn', 'Actual: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
    ))
    fig.update_layout(
        title="Confusion Matrix",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"✅ **True Negatives**: {cm[0,0]:,} customers correctly predicted as staying")
        st.error(f"❌ **False Positives**: {cm[0,1]:,} customers incorrectly predicted as churning")
    
    with col2:
        st.error(f"❌ **False Negatives**: {cm[1,0]:,} customers incorrectly predicted as staying")
        st.success(f"✅ **True Positives**: {cm[1,1]:,} customers correctly predicted as churning")

# ========== PAGE 4: BUSINESS IMPACT ==========
else:
    st.markdown('<p class="main-header">📈 Business Impact Analysis</p>', unsafe_allow_html=True)
    
    st.subheader("💰 ROI Calculator")
    
    # Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        avg_clv = st.number_input("Average Customer Lifetime Value ($)", value=2000, step=100)
        intervention_cost = st.number_input("Cost per Intervention ($)", value=50, step=10)
    
    with col2:
        success_rate = st.slider("Intervention Success Rate (%)", 0, 100, 30) / 100
        total_customer_base = st.number_input("Total Customer Base", value=100000, step=10000)
    
    st.markdown("---")
    
    # Calculations
    high_risk_count = (y_pred_proba > 0.7).sum()
    scaling_factor = total_customer_base / len(X_test)
    
    projected_high_risk = int(high_risk_count * scaling_factor)
    customers_saved = int(projected_high_risk * success_rate)
    total_intervention_cost = projected_high_risk * intervention_cost
    revenue_saved = customers_saved * avg_clv
    net_benefit = revenue_saved - total_intervention_cost
    roi_percentage = (net_benefit / total_intervention_cost * 100) if total_intervention_cost > 0 else 0
    
    # Display results
    st.subheader("📊 Projected Annual Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "High-Risk Customers",
            f"{projected_high_risk:,}",
            delta=f"{projected_high_risk/total_customer_base*100:.1f}% of base"
        )
    
    with col2:
        st.metric(
            "Customers Saved",
            f"{customers_saved:,}",
            delta=f"{success_rate*100:.0f}% success rate"
        )
    
    with col3:
        st.metric(
            "Intervention Cost",
            f"${total_intervention_cost:,}",
            delta=f"${intervention_cost}/customer"
        )
    
    with col4:
        st.metric(
            "Revenue Saved",
            f"${revenue_saved:,}",
            delta=f"${avg_clv}/customer"
        )
    
    # Net benefit
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💵 Net Benefit")
        st.markdown(f'<p style="font-size: 48px; font-weight: bold; color: {"green" if net_benefit > 0 else "red"};">${net_benefit:,}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 ROI")
        st.markdown(f'<p style="font-size: 48px; font-weight: bold; color: {"green" if roi_percentage > 0 else "red"};">{roi_percentage:.0f}%</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Financial Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost vs Revenue
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Cost',
            x=['Intervention'],
            y=[total_intervention_cost],
            marker_color='red'
        ))
        fig.add_trace(go.Bar(
            name='Revenue Saved',
            x=['Intervention'],
            y=[revenue_saved],
            marker_color='green'
        ))
        fig.update_layout(
            title="Cost vs Revenue Saved",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer flow
        labels = ['Total Customers', 'High Risk', 'Customers Saved', 'Revenue Generated']
        values = [total_customer_base, projected_high_risk, customers_saved, revenue_saved/avg_clv]
        
        fig = go.Figure(data=[go.Funnel(
            y=labels,
            x=values,
            textinfo="value+percent initial"
        )])
        fig.update_layout(
            title="Customer Retention Funnel",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    if roi_percentage > 100:
        st.success("✅ **Excellent ROI!** The churn prevention program is highly profitable.")
        st.info("💡 Consider expanding intervention programs and increasing budget allocation.")
    elif roi_percentage > 0:
        st.warning("⚠️ **Positive but modest ROI.** Room for improvement.")
        st.info("💡 Focus on improving intervention success rate through better targeting and personalization.")
    else:
        st.error("❌ **Negative ROI.** Current approach needs optimization.")
        st.info("💡 Reduce intervention costs or improve success rate. Consider more targeted approaches.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Customer Churn Prediction System | Built with Streamlit, ML, and SHAP</p>
        <p>Model: Random Forest | ROC AUC: 0.835 | Data: Telco Customer Churn</p>
    </div>
""", unsafe_allow_html=True)