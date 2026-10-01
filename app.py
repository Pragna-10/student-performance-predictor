"""
AI-Based Student Performance Prediction System
Built with Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# Import local modules
from styles import CUSTOM_CSS, setup_matplotlib_theme
from data_generator import generate_student_dataset, save_default_datasets
from model_utils import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    train_and_evaluate_regression,
    train_and_evaluate_classification,
    predict_single_student,
    generate_actionable_insights,
    calculate_sensitivity_curve
)

# Page configuration
st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling & matplotlib aesthetics
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
setup_matplotlib_theme()

# Paths & Initial Directory Setup
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "students_data.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_batch_roster.csv")

# Ensure dataset exists on startup
if not os.path.exists(DEFAULT_DATASET_PATH) or not os.path.exists(SAMPLE_BATCH_PATH):
    save_default_datasets(DATA_DIR)

# Cache dataset loading
@st.cache_data
def load_data(file_path_or_buffer):
    return pd.read_csv(file_path_or_buffer)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 10px; border-radius: 12px; font-size: 24px;">🎓</div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #f8fafc;">EduPredict AI</h3>
                <span style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Early Warning & Analytics</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Data Source")
    data_source_mode = st.radio(
        "Choose Dataset Source:",
        ["Standard Benchmark (1,500 Students)", "Upload Custom CSV"],
        index=0,
        label_visibility="collapsed"
    )
    
    custom_uploaded_file = None
    if data_source_mode == "Upload Custom CSV":
        custom_uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV with student attributes and Final_Exam_Score"
        )
        
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_choice = st.selectbox(
        "Primary ML Algorithm:",
        [
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Linear Regression"
        ],
        index=0
    )
    
    test_split = st.slider("Validation Test Split:", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
    
    with st.expander("🛠️ Advanced Hyperparameters"):
        rf_trees = st.slider("Number of Trees (RF / GB):", 50, 300, 100, step=25)
        rf_depth = st.slider("Max Tree Depth:", 3, 20, 10)
        gb_lr = st.select_slider("Gradient Boosting Learning Rate:", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.8rem; color: #94a3b8;">
            <b style="color: #cbd5e1;">System Stack:</b><br/>
            • Scikit-learn Pipeline<br/>
            • Pandas & NumPy Preprocessing<br/>
            • Matplotlib Visualizations<br/>
            • Streamlit Dynamic UI
        </div>
    """, unsafe_allow_html=True)

# ----------------- DATA LOADING & VALIDATION -----------------
try:
    if custom_uploaded_file is not None:
        df = load_data(custom_uploaded_file)
        st.sidebar.success(f"Loaded custom file: {len(df)} records")
    else:
        df = load_data(DEFAULT_DATASET_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}. Generating fallback dataset...")
    df = generate_student_dataset(n_samples=1200)

# Quick validation check
missing_num = [c for c in NUMERIC_FEATURES if c not in df.columns]
if missing_num:
    st.error(f"Missing required numeric columns in dataset: {missing_num}")
    st.stop()

# ----------------- CACHED MODEL TRAINING -----------------
@st.cache_resource(show_spinner=False)
def get_trained_model(df_hash_data: pd.DataFrame, model_name: str, test_size: float, n_trees: int, depth: int, lr: float):
    return train_and_evaluate_regression(
        df=df_hash_data,
        model_name=model_name,
        test_size=test_size,
        rf_estimators=n_trees,
        rf_depth=depth,
        gb_estimators=n_trees,
        learning_rate=lr
    )

with st.spinner("Optimizing and training AI models..."):
    regression_results = get_trained_model(df, model_choice, test_split, rf_trees, rf_depth, gb_lr)
    active_pipeline = regression_results["pipeline"]

# ----------------- HERO HEADER -----------------
st.markdown("""
    <div class="hero-header">
        <div style="display: flex; gap: 8px; margin-bottom: 8px;">
            <span class="badge-pill badge-ai">● AI-Driven Early Warning Engine</span>
            <span class="badge-pill badge-green">Production Ready</span>
        </div>
        <h1 class="hero-title">Student Performance & Risk Intelligence System</h1>
        <p class="hero-subtitle">
            Harness machine learning to accurately forecast student academic outcomes, identify high-risk students before final examinations, and prescribe targeted, personalized pedagogical interventions.
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------- APPLICATION TABS -----------------
tab_eda, tab_models, tab_predict, tab_batch, tab_whatif = st.tabs([
    "📊 Exploratory Analytics",
    "🤖 ML Benchmark & Diagnostics",
    "🎯 Individual Student Predictor",
    "📈 Batch Risk Radar",
    "🧪 What-If Sensitivity Simulator"
])

# ==============================================================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown('<div class="section-title">📊 Cohort Overview & KPI Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Key diagnostic metrics across the selected academic cohort.</div>', unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_students = len(df)
    avg_score = df[TARGET_REGRESSION].mean()
    pass_count = (df[TARGET_REGRESSION] >= 50.0).sum()
    pass_rate = (pass_count / total_students) * 100
    avg_study = df["Study_Hours_Per_Week"].mean()
    at_risk_count = (df[TARGET_REGRESSION] < 50.0).sum()
    at_risk_rate = (at_risk_count / total_students) * 100
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Cohort</div>
                <div class="metric-value">{total_students:,}</div>
                <div class="metric-sub">Active Enrolled Students</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Exam Score</div>
                <div class="metric-value">{avg_score:.1f}</div>
                <div class="metric-sub">Scale: 0 - 100</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pass Rate</div>
                <div class="metric-value" style="color: #34d399;">{pass_rate:.1f}%</div>
                <div class="metric-sub">{pass_count} Passed (&ge; 50 pts)</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">At-Risk Count</div>
                <div class="metric-value" style="color: #f87171;">{at_risk_count}</div>
                <div class="metric-sub">{at_risk_rate:.1f}% flagged for support</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Study Hours</div>
                <div class="metric-value">{avg_study:.1f}h</div>
                <div class="metric-sub">Weekly study dedication</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        st.markdown("#### 📈 Exam Score Distribution & Risk Thresholds")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        
        scores = df[TARGET_REGRESSION]
        n, bins, patches = ax.hist(scores, bins=25, edgecolor="#1f2937", alpha=0.85)
        
        # Color bins based on thresholds
        for bin_left, patch in zip(bins[:-1], patches):
            if bin_left < 50:
                patch.set_facecolor('#ef4444')  # Red (At risk)
            elif bin_left < 65:
                patch.set_facecolor('#f59e0b')  # Amber (Average)
            elif bin_left < 80:
                patch.set_facecolor('#3b82f6')  # Blue (Good)
            else:
                patch.set_facecolor('#10b981')  # Emerald (Excellent)
                
        # Vertical reference lines
        ax.axvline(avg_score, color='#ffffff', linestyle='--', linewidth=1.8, label=f'Mean Score ({avg_score:.1f})')
        ax.axvline(50.0, color='#f87171', linestyle=':', linewidth=1.8, label='Passing Threshold (50)')
        
        ax.set_xlabel("Final Exam Score")
        ax.set_ylabel("Student Count")
        ax.legend(frameon=True, facecolor="#1f2937", edgecolor="#374151")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.markdown("#### 🔍 Correlation Matrix (Academic Factors vs Score)")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        
        corr_cols = NUMERIC_FEATURES + [TARGET_REGRESSION]
        corr_matrix = df[corr_cols].corr()
        
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
        labels_clean = [c.replace("_", " ") for c in corr_cols]
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(labels_clean, rotation=40, ha="left", fontsize=8)
        ax.set_yticklabels(labels_clean, fontsize=8)
        
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color="white" if abs(val) > 0.4 else "black", fontsize=8, weight='bold')
                
        ax.grid(False)
        st.pyplot(fig)
        plt.close(fig)

    chart_col3, chart_col4 = st.columns([1, 1])
    
    with chart_col3:
        st.markdown("#### 🎯 Study Hours vs Final Score (Color: Attendance)")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        
        scatter = ax.scatter(
            df["Study_Hours_Per_Week"],
            df[TARGET_REGRESSION],
            c=df["Attendance_Rate"],
            cmap="viridis",
            alpha=0.75,
            edgecolors='none',
            s=35
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attendance Rate (%)', color="#9CA3AF")
        
        # Best fit polynomial trendline
        p = np.poly1d(np.polyfit(df["Study_Hours_Per_Week"], df[TARGET_REGRESSION], 2))
        x_trend = np.linspace(df["Study_Hours_Per_Week"].min(), df["Study_Hours_Per_Week"].max(), 100)
        ax.plot(x_trend, p(x_trend), color="#f43f5e", linewidth=2.5, label="Trend Curve")
        
        ax.set_xlabel("Weekly Study Hours")
        ax.set_ylabel("Final Exam Score")
        ax.legend(frameon=True, facecolor="#1f2937", edgecolor="#374151")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col4:
        st.markdown("#### 👥 Impact of Tutoring & Parental Education")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        
        edu_order = ["High School", "Associate Degree", "Bachelor's Degree", "Master's / PhD"]
        grouped = df.groupby(["Parental_Education", "Tutoring"])[TARGET_REGRESSION].mean().unstack()
        grouped = grouped.reindex(edu_order)
        
        x = np.arange(len(edu_order))
        width = 0.35
        
        tutoring_no = grouped["No"] if "No" in grouped.columns else [0]*len(edu_order)
        tutoring_yes = grouped["Yes"] if "Yes" in grouped.columns else [0]*len(edu_order)
        
        ax.bar(x - width/2, tutoring_no, width, label='No Tutoring', color='#64748b', alpha=0.9)
        ax.bar(x + width/2, tutoring_yes, width, label='With Tutoring', color='#6366f1', alpha=0.9)
        
        ax.set_ylabel("Average Exam Score")
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace(" Degree", "") for e in edu_order], rotation=15, ha="right")
        ax.set_ylim(0, 100)
        ax.legend(frameon=True, facecolor="#1f2937", edgecolor="#374151")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Dataset Inspection
    with st.expander("📋 View Raw Student Dataset (First 100 Records & Descriptive Stats)"):
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown("##### 📐 Summary Statistics")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

# ==============================================================================
# TAB 2: MODEL TRAINING & COMPARISON CENTER
# ==============================================================================
with tab_models:
    st.markdown('<div class="section-title">🤖 Machine Learning Leaderboard & Model Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Live performance comparison across multiple regression & classification architectures.</div>', unsafe_allow_html=True)
    
    # Run multi-model benchmark
    with st.spinner("Benchmarking regression algorithms..."):
        benchmark_models = ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression", "Linear Regression"]
        benchmark_results = []
        
        for m_name in benchmark_models:
            res = get_trained_model(df, m_name, test_split, rf_trees, rf_depth, gb_lr)
            row = {"Model": m_name}
            row.update(res["metrics"])
            benchmark_results.append(row)
            
        bench_df = pd.DataFrame(benchmark_results)

    # Benchmark display
    col_bench1, col_bench2 = st.columns([1.1, 0.9])
    with col_bench1:
        st.markdown("#### 🏆 Regression Algorithms Benchmark")
        # Format highlight table
        st.dataframe(
            bench_df.style.highlight_max(subset=["R2 Score"], color="#1e3a5f")
                          .highlight_min(subset=["RMSE", "MAE"], color="#14532d"),
            use_container_width=True
        )
        st.caption("Lower RMSE and MAE indicate superior precision; Higher R² Score indicates better variance explanation.")

    with col_bench2:
        st.markdown("#### 📊 Comparative R² Score vs RMSE")
        fig, ax1 = plt.subplots(figsize=(6, 3.2), dpi=120)
        
        model_names = [m.replace(" Regressor", "").replace(" Regression", "") for m in bench_df["Model"]]
        x = np.arange(len(model_names))
        
        color1 = '#6366f1'
        ax1.set_ylabel('R² Score', color=color1)
        b1 = ax1.bar(x - 0.2, bench_df["R2 Score"], 0.4, color=color1, alpha=0.85, label='R² Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.05)
        
        ax2 = ax1.twinx()
        color2 = '#f43f5e'
        ax2.set_ylabel('RMSE (Error)', color=color2)
        b2 = ax2.bar(x + 0.2, bench_df["RMSE"], 0.4, color=color2, alpha=0.85, label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown(f"### 🔬 Deep Dive: Selected Model — `{model_choice}`")
    
    diag_col1, diag_col2, diag_col3 = st.columns([1, 1, 1.2])
    
    with diag_col1:
        st.markdown("##### 🎯 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=120)
        y_test = regression_results["y_test"]
        y_pred = regression_results["y_pred"]
        
        ax.scatter(y_test, y_pred, alpha=0.6, color="#818cf8", edgecolors='none', s=25)
        # 45 degree perfect prediction line
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], color="#34d399", linestyle="--", linewidth=2, label="Ideal (y = x)")
        
        ax.set_xlabel("Actual Exam Score")
        ax.set_ylabel("Predicted Exam Score")
        ax.legend(facecolor="#1f2937", edgecolor="#374151")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with diag_col2:
        st.markdown("##### 📉 Residuals Error Distribution")
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=120)
        residuals = regression_results["residuals"]
        
        ax.hist(residuals, bins=20, color="#ec4899", edgecolor="#1f2937", alpha=0.8)
        ax.axvline(0, color="#ffffff", linestyle="--", linewidth=1.8, label="Zero Error")
        
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.legend(facecolor="#1f2937", edgecolor="#374151")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with diag_col3:
        st.markdown("##### 💡 Global Feature Importance")
        feat_df = regression_results["feature_importance"]
        if not feat_df.empty:
            top_feats = feat_df.head(8).iloc[::-1]  # reversed for horizontal plot
            fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
            
            bars = ax.barh(top_feats["Clean_Feature"], top_feats["Weight (%)"], color="#38bdf8", alpha=0.85)
            ax.set_xlabel("Relative Weight / Importance (%)")
            
            # Label bars
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                        va='center', ha='left', color='#e2e8f0', fontsize=8, weight='bold')
                
            ax.set_xlim(0, max(top_feats["Weight (%)"]) * 1.25)
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature importance is not directly exposed by this linear model architecture.")

    # Classification Benchmark
    with st.expander("🏷️ View Multi-Class Student Risk Classification Benchmark"):
        with st.spinner("Training classification models..."):
            clf_rf = train_and_evaluate_classification(df, "Random Forest Classifier", test_split, rf_trees, rf_depth)
            clf_gb = train_and_evaluate_classification(df, "Gradient Boosting Classifier", test_split, rf_trees, rf_depth)
            clf_lr = train_and_evaluate_classification(df, "Logistic Regression", test_split)
            
            clf_bench_df = pd.DataFrame([
                {"Model": "Random Forest Classifier", **clf_rf["metrics"]},
                {"Model": "Gradient Boosting Classifier", **clf_gb["metrics"]},
                {"Model": "Logistic Regression", **clf_lr["metrics"]}
            ])
            st.dataframe(clf_bench_df, use_container_width=True)
            
            # Confusion matrix of RF
            st.markdown("##### Confusion Matrix (Random Forest Classifier)")
            fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=120)
            cm = clf_rf["confusion_matrix"]
            classes = clf_rf["classes"]
            
            cax = ax.matshow(cm, cmap='Blues')
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=20, ha='left', fontsize=8)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel("Predicted Class", labelpad=10)
            ax.set_ylabel("True Class")
            
            for i in range(len(classes)):
                for j in range(len(classes)):
                    val = cm[i, j]
                    ax.text(j, i, str(val), ha='center', va='center',
                            color="white" if val > cm.max()/2 else "black", weight='bold')
                    
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# TAB 3: INDIVIDUAL STUDENT PREDICTOR
# ==============================================================================
with tab_predict:
    st.markdown('<div class="section-title">🎯 Single Student Performance Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Adjust student parameters below to generate real-time AI performance predictions and intervention recommendations.</div>', unsafe_allow_html=True)
    
    col_input, col_output = st.columns([1.3, 1.0])
    
    with col_input:
        st.markdown("#### 1. Academic & Behavioral Indicators")
        inp_col1, inp_col2 = st.columns(2)
        
        with inp_col1:
            p_past_score = st.slider(
                "Past Exam Score (0 - 100):",
                min_value=20.0, max_value=100.0, value=68.0, step=0.5,
                help="Historical average in prerequisite exams"
            )
            p_study_hours = st.slider(
                "Weekly Study Hours:",
                min_value=1.0, max_value=40.0, value=12.0, step=0.5,
                help="Dedicated self-study hours outside class"
            )
            p_sleep_hours = st.slider(
                "Nightly Sleep Hours:",
                min_value=4.0, max_value=10.0, value=7.0, step=0.5,
                help="Average nightly sleep duration"
            )
            
        with inp_col2:
            p_attendance = st.slider(
                "Class Attendance Rate (%):",
                min_value=40.0, max_value=100.0, value=82.0, step=1.0,
                help="Lecture and lab attendance percentage"
            )
            p_assignments = st.slider(
                "Assignment Completion (%):",
                min_value=30.0, max_value=100.0, value=85.0, step=1.0,
                help="Homework and project submission compliance"
            )
            p_stress = st.selectbox(
                "Perceived Stress Level:",
                ["Low", "Medium", "High"],
                index=1
            )
            
        st.markdown("#### 2. Environmental & Support Profile")
        env_col1, env_col2 = st.columns(2)
        
        with env_col1:
            p_parent_edu = st.selectbox(
                "Parental Education:",
                ["High School", "Associate Degree", "Bachelor's Degree", "Master's / PhD"],
                index=2
            )
            p_study_env = st.selectbox(
                "Home Study Environment:",
                ["Quiet / Dedicated", "Moderate Noise", "Noisy / Distracting"],
                index=0
            )
            
        with env_col2:
            p_tutoring = st.radio("Enrolled in Tutoring?", ["Yes", "No"], index=1, horizontal=True)
            p_internet = st.radio("High-Speed Internet?", ["Yes", "No"], index=0, horizontal=True)
            p_extra = st.radio("Extracurricular Activities?", ["Yes", "No"], index=0, horizontal=True)
            
    # Student Dictionary
    student_record = {
        "Study_Hours_Per_Week": p_study_hours,
        "Attendance_Rate": p_attendance,
        "Past_Exam_Score": p_past_score,
        "Sleep_Hours": p_sleep_hours,
        "Assignments_Completion_Rate": p_assignments,
        "Parental_Education": p_parent_edu,
        "Internet_Access": p_internet,
        "Tutoring": p_tutoring,
        "Extracurricular_Activities": p_extra,
        "Study_Environment": p_study_env,
        "Stress_Level": p_stress
    }
    
    # Run Prediction
    pred_score, pred_cat, pred_risk = predict_single_student(active_pipeline, student_record)
    
    # Recommendation insights
    recommendations = generate_actionable_insights(student_record, pred_score, active_pipeline)
    
    with col_output:
        st.markdown("#### 🔮 AI Predicted Outcome")
        
        # Color styling
        if pred_score >= 80:
            score_color_class = "score-green"
            badge_class = "badge-green"
            verdict_desc = "Student demonstrates strong mastery and is on track for honors."
        elif pred_score >= 65:
            score_color_class = "score-green"
            badge_class = "badge-ai"
            verdict_desc = "Student performs reliably with solid academic retention."
        elif pred_score >= 50:
            score_color_class = "score-amber"
            badge_class = "badge-amber"
            verdict_desc = "Student is near borderline threshold; proactive support advised."
        else:
            score_color_class = "score-red"
            badge_class = "badge-rose"
            verdict_desc = "CRITICAL ALERT: Student is at high risk of course failure without intervention."

        st.markdown(f"""
            <div class="prediction-score-display">
                <div style="margin-bottom: 0.5rem;">
                    <span class="badge-pill {badge_class}">Risk Level: {pred_risk}</span>
                </div>
                <div class="score-number {score_color_class}">{pred_score:.1f}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                    {pred_cat}
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 320px; margin: 0 auto;">
                    {verdict_desc}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Actionable recommendations
        st.markdown("#### 💡 Prescriptive Intervention Plan")
        for rec in recommendations:
            st.markdown(f"""
                <div class="rec-box">
                    <div style="font-weight: 700; display: flex; align-items: center; justify-content: space-between; margin-bottom: 3px;">
                        <span>{rec['icon']} {rec['category']}</span>
                        <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">{rec['impact']}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #cbd5e1;">{rec['action']}</div>
                </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: BATCH PREDICTION & STUDENT RISK RADAR
# ==============================================================================
with tab_batch:
    st.markdown('<div class="section-title">📈 Batch Evaluation & Early Warning Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Score an entire classroom or cohort at once to generate triage lists and exportable reports.</div>', unsafe_allow_html=True)
    
    col_batch_upload, col_batch_actions = st.columns([2, 1])
    
    with col_batch_upload:
        batch_file = st.file_uploader(
            "Upload Batch Student Roster (CSV):",
            type=["csv"],
            key="batch_roster_uploader",
            help="Upload a CSV containing student records without final grades"
        )
        
    with col_batch_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample_btn = st.button("📥 Load Built-In Sample Roster (40 Students)")
        
    # Determine batch DataFrame
    batch_df = None
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
    elif use_sample_btn or os.path.exists(SAMPLE_BATCH_PATH):
        batch_df = pd.read_csv(SAMPLE_BATCH_PATH)
        
    if batch_df is not None:
        # Run batch predictions
        with st.spinner("Scoring student cohort..."):
            preds = active_pipeline.predict(batch_df)
            preds = np.clip(preds, 0.0, 100.0).round(1)
            
            scored_batch = batch_df.copy()
            scored_batch["Predicted_Exam_Score"] = preds
            
            # Classifications
            conditions_cat = [
                scored_batch["Predicted_Exam_Score"] >= 80.0,
                (scored_batch["Predicted_Exam_Score"] >= 65.0) & (scored_batch["Predicted_Exam_Score"] < 80.0),
                (scored_batch["Predicted_Exam_Score"] >= 50.0) & (scored_batch["Predicted_Exam_Score"] < 65.0),
                scored_batch["Predicted_Exam_Score"] < 50.0
            ]
            choices_cat = ["Excellent (A)", "Good (B)", "Average (C)", "At-Risk (Fail)"]
            scored_batch["Predicted_Category"] = np.select(conditions_cat, choices_cat, default="Average (C)")
            
            conditions_risk = [
                scored_batch["Predicted_Exam_Score"] < 50.0,
                (scored_batch["Predicted_Exam_Score"] >= 50.0) & (scored_batch["Predicted_Exam_Score"] < 65.0),
                scored_batch["Predicted_Exam_Score"] >= 65.0
            ]
            choices_risk = ["High Risk", "Moderate Risk", "Low Risk"]
            scored_batch["Risk_Level"] = np.select(conditions_risk, choices_risk, default="Moderate Risk")

        # KPI Summaries
        bkpi1, bkpi2, bkpi3, bkpi4 = st.columns(4)
        total_batch = len(scored_batch)
        high_risk_batch = (scored_batch["Risk_Level"] == "High Risk").sum()
        mod_risk_batch = (scored_batch["Risk_Level"] == "Moderate Risk").sum()
        avg_batch_score = scored_batch["Predicted_Exam_Score"].mean()
        
        with bkpi1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Evaluated Cohort</div>
                    <div class="metric-value">{total_batch}</div>
                    <div class="metric-sub">Roster Size</div>
                </div>
            """, unsafe_allow_html=True)
            
        with bkpi2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Predicted Score</div>
                    <div class="metric-value">{avg_batch_score:.1f}</div>
                    <div class="metric-sub">Overall Class Average</div>
                </div>
            """, unsafe_allow_html=True)
            
        with bkpi3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High-Risk Flagged</div>
                    <div class="metric-value" style="color: #f87171;">{high_risk_batch}</div>
                    <div class="metric-sub">Require immediate support</div>
                </div>
            """, unsafe_allow_html=True)
            
        with bkpi4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Moderate Risk</div>
                    <div class="metric-value" style="color: #fbbf24;">{mod_risk_batch}</div>
                    <div class="metric-sub">Watchlist students</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Donut Chart & High Risk Alert List
        col_donut, col_alert_list = st.columns([1, 1.4])
        
        with col_donut:
            st.markdown("#### 🍩 Risk Composition Breakdown")
            fig, ax = plt.subplots(figsize=(5, 3.8), dpi=120)
            
            risk_counts = scored_batch["Risk_Level"].value_counts()
            colors_map = {
                "High Risk": "#ef4444",
                "Moderate Risk": "#f59e0b",
                "Low Risk": "#10b981"
            }
            colors = [colors_map.get(k, "#6366f1") for k in risk_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                risk_counts,
                labels=risk_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                pctdistance=0.75,
                wedgeprops=dict(width=0.45, edgecolor='#111827')
            )
            for t in texts:
                t.set_color('#e2e8f0')
                t.set_fontsize(9)
            for at in autotexts:
                at.set_color('#ffffff')
                at.set_weight('bold')
                at.set_fontsize(9)
                
            st.pyplot(fig)
            plt.close(fig)

        with col_alert_list:
            st.markdown("#### 🚨 Priority Intervention Watchlist (High Risk Students)")
            high_risk_df = scored_batch[scored_batch["Risk_Level"] == "High Risk"][
                ["Student_ID", "Past_Exam_Score", "Attendance_Rate", "Study_Hours_Per_Week", "Predicted_Exam_Score"]
            ] if "Student_ID" in scored_batch.columns else scored_batch[scored_batch["Risk_Level"] == "High Risk"]
            
            if not high_risk_df.empty:
                st.dataframe(high_risk_df, use_container_width=True)
            else:
                st.success("🎉 Excellent! No students in this batch are classified in the High-Risk category.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Full Scored Roster Table
        st.markdown("#### 📋 Complete Scored Cohort Roster")
        st.dataframe(scored_batch, use_container_width=True)
        
        # Export CSV Button
        csv_buffer = io.StringIO()
        scored_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Scored Report as CSV",
            data=csv_buffer.getvalue(),
            file_name="student_performance_predictions_report.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a student CSV file or click 'Load Built-In Sample Roster' to run batch prediction.")

# ==============================================================================
# TAB 5: WHAT-IF SENSITIVITY SIMULATOR
# ==============================================================================
with tab_whatif:
    st.markdown('<div class="section-title">🧪 "What-If" Sensitivity Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Dynamically observe how modifying a single academic or behavioral habit shifts predicted exam performance, keeping all other variables constant.</div>', unsafe_allow_html=True)
    
    col_sim_controls, col_sim_plot = st.columns([1, 1.6])
    
    with col_sim_controls:
        st.markdown("#### Simulation Baseline")
        sim_feature = st.selectbox(
            "Feature to Perturb & Analyze:",
            [
                "Study_Hours_Per_Week",
                "Attendance_Rate",
                "Sleep_Hours",
                "Assignments_Completion_Rate",
                "Past_Exam_Score"
            ],
            index=0
        )
        
        st.markdown("##### Baseline Student Archetype:")
        base_study = st.slider("Baseline Study Hours:", 2.0, 35.0, 10.0, step=1.0, key="sim_base_study")
        base_attend = st.slider("Baseline Attendance Rate (%):", 50.0, 100.0, 75.0, step=1.0, key="sim_base_attend")
        base_past = st.slider("Baseline Past Score:", 30.0, 95.0, 65.0, step=1.0, key="sim_base_past")
        base_sleep = st.slider("Baseline Sleep Hours:", 4.0, 10.0, 7.0, step=0.5, key="sim_base_sleep")
        base_assign = st.slider("Baseline Assignments (%):", 40.0, 100.0, 80.0, step=1.0, key="sim_base_assign")
        
    sim_base_record = {
        "Study_Hours_Per_Week": base_study,
        "Attendance_Rate": base_attend,
        "Past_Exam_Score": base_past,
        "Sleep_Hours": base_sleep,
        "Assignments_Completion_Rate": base_assign,
        "Parental_Education": "Bachelor's Degree",
        "Internet_Access": "Yes",
        "Tutoring": "No",
        "Extracurricular_Activities": "Yes",
        "Study_Environment": "Quiet / Dedicated",
        "Stress_Level": "Medium"
    }

    # Generate simulation curve
    ranges = {
        "Study_Hours_Per_Week": np.linspace(1, 38, 50),
        "Attendance_Rate": np.linspace(40, 100, 50),
        "Sleep_Hours": np.linspace(4.0, 10.0, 40),
        "Assignments_Completion_Rate": np.linspace(30, 100, 50),
        "Past_Exam_Score": np.linspace(30, 99, 50)
    }
    
    curve_df = calculate_sensitivity_curve(
        active_pipeline,
        sim_base_record,
        sim_feature,
        ranges[sim_feature]
    )

    with col_sim_plot:
        st.markdown(f"#### 📈 Impact Curve: `{sim_feature.replace('_', ' ')}` vs Predicted Score")
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
        
        x_vals = curve_df[sim_feature]
        y_vals = curve_df["Predicted_Score"]
        
        ax.plot(x_vals, y_vals, color="#818cf8", linewidth=3.0, label="Predicted Score Curve")
        ax.fill_between(x_vals, y_vals, alpha=0.15, color="#818cf8")
        
        # Current baseline point marker
        cur_x = sim_base_record[sim_feature]
        cur_y = float(active_pipeline.predict(pd.DataFrame([sim_base_record]))[0])
        ax.scatter([cur_x], [cur_y], color="#f43f5e", s=130, zorder=5, label=f"Current Baseline ({cur_x:.1f}, {cur_y:.1f})")
        
        # Risk thresholds
        ax.axhline(50, color="#ef4444", linestyle=":", alpha=0.7, label="Passing Line (50)")
        ax.axhline(80, color="#10b981", linestyle=":", alpha=0.7, label="Honor Line (80)")
        
        ax.set_xlabel(sim_feature.replace('_', ' '), fontsize=11)
        ax.set_ylabel("Predicted Final Exam Score", fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(facecolor="#1f2937", edgecolor="#374151", loc="upper left")
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Analytical Takeaway
        gain = y_vals.max() - y_vals.min()
        st.markdown(f"""
            <div class="rec-box">
                <b>💡 Sensitivity Observation:</b><br/>
                Modulating <code>{sim_feature.replace('_', ' ')}</code> across its spectrum creates an expected performance variance of <b>+{gain:.1f} points</b>.
                Notice non-linear dynamics, diminishing returns, or inflection zones along the trajectory.
            </div>
        """, unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0;">
        <b>AI-Based Student Performance Prediction System</b> • Built with Python, Pandas, NumPy, Scikit-learn, Matplotlib & Streamlit.<br/>
        Equipping educators, academic advisors, and students with transparent, proactive predictive intelligence.
    </div>
""", unsafe_allow_html=True)
