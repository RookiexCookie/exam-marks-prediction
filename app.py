import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from utils.data_generator import (
    load_or_create_dataset,
    generate_synthetic_data,
    clean_and_validate_dataframe,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    DEFAULT_CSV_PATH,
)
from utils.model_utils import (
    AVAILABLE_MODELS,
    train_and_eval_model,
    compare_all_models,
    predict_exam_marks,
    get_feature_importance_df,
    generate_student_recommendations,
    simulate_target_goal,
    get_grade_info,
)
from utils.styles import CUSTOM_CSS, render_hero, render_prediction_badge

# Page Configuration
st.set_page_config(
    page_title="Exam Marks Prediction | ML Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_or_create_dataset()

if "test_size" not in st.session_state:
    st.session_state.test_size = 0.20

if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "Random Forest Regressor"

# Cache-friendly model trainer
@st.cache_resource(show_spinner=False)
def get_trained_models(df: pd.DataFrame, test_size: float, random_seed: int):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    leaderboard, models_map = compare_all_models(X_train, y_train, X_test, y_test)
    return leaderboard, models_map, X_train, X_test, y_train, y_test

leaderboard_df, models_map, X_train, X_test, y_train, y_test = get_trained_models(
    st.session_state.df,
    st.session_state.test_size,
    st.session_state.random_seed
)

# Current active model
active_model_res = models_map.get(
    st.session_state.selected_model_name,
    models_map["Linear Regression"]
)

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 **Exam Marks AI**")
    st.caption("Machine Learning Academic Predictor")
    st.markdown("---")

    nav_selection = st.radio(
        "**Navigation**",
        [
            "📊 Dashboard Overview",
            "🔮 Predict Exam Marks",
            "🎯 Target Marks Planner",
            "⚙️ Model Training & Evaluation",
            "💡 Insights & Recommendations",
            "📁 Dataset Explorer"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ **Active Model**")
    selected_model = st.selectbox(
        "Predictive Engine",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_name),
        help="Select which algorithm powers the predictions and feature importance analysis."
    )
    if selected_model != st.session_state.selected_model_name:
        st.session_state.selected_model_name = selected_model
        st.rerun()

    active_model_res = models_map[st.session_state.selected_model_name]
    st.caption(f"**R² Score:** `{active_model_res['r2']:.4f}` | **RMSE:** `{active_model_res['rmse']:.2f}`")

    st.markdown("---")
    st.markdown("### 📌 **Quick Info**")
    st.markdown(
        f"""
        - **Total Records:** `{len(st.session_state.df):,}`
        - **Features Used:** `{len(FEATURE_COLUMNS)}`
        - **Class Pass Rate:** `{(st.session_state.df[TARGET_COLUMN] >= 40).mean() * 100:.1f}%`
        - **Mean Exam Mark:** `{st.session_state.df[TARGET_COLUMN].mean():.1f} / 100`
        """
    )
    st.markdown("---")
    st.caption("College ML Mini-Project • Built with Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# 1. DASHBOARD OVERVIEW
# ---------------------------------------------------------
if nav_selection == "📊 Dashboard Overview":
    st.markdown(
        render_hero(
            "🎓 Student Exam Marks Prediction Dashboard",
            "Predict student examination performance using machine learning regression. Analyze historical study habits, attendance trends, and continuous assessment metrics to optimize academic outcomes."
        ),
        unsafe_allow_html=True
    )

    # KPI Top Bar
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Cohort", f"{len(st.session_state.df):,} students")
    with kpi2:
        st.metric("Average Exam Mark", f"{st.session_state.df[TARGET_COLUMN].mean():.1f} / 100")
    with kpi3:
        pass_pct = (st.session_state.df[TARGET_COLUMN] >= 40).mean() * 100
        st.metric("Passing Rate", f"{pass_pct:.1f}%", delta=f"{pass_pct - 75:.1f}% vs Goal")
    with kpi4:
        st.metric("Avg Daily Study", f"{st.session_state.df['study_hours'].mean():.1f} hrs/day")
    with kpi5:
        st.metric("Avg Attendance", f"{st.session_state.df['attendance_percentage'].mean():.1f}%")

    st.markdown("### 📈 **Performance & Factor Visualizations**")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Distribution of Exam Marks
        fig_dist = px.histogram(
            st.session_state.df,
            x=TARGET_COLUMN,
            nbins=28,
            title="Distribution of Examination Marks",
            labels={TARGET_COLUMN: "Exam Marks (/ 100)"},
            color_discrete_sequence=["#6366F1"],
            marginal="box"
        )
        fig_dist.add_vline(x=40, line_dash="dash", line_color="#EF4444", annotation_text="Pass Mark (40)")
        fig_dist.add_vline(x=75, line_dash="dot", line_color="#10B981", annotation_text="Distinction (75)")
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        # Study Hours vs Exam Marks
        fig_study = px.scatter(
            st.session_state.df,
            x="study_hours",
            y=TARGET_COLUMN,
            trendline="ols",
            color="attendance_percentage",
            color_continuous_scale="Viridis",
            title="Study Hours vs Exam Marks (Colored by Attendance %)",
            labels={
                "study_hours": "Daily Study Hours",
                TARGET_COLUMN: "Exam Marks (/ 100)",
                "attendance_percentage": "Attendance %"
            },
            opacity=0.65
        )
        fig_study.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_study, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Attendance vs Exam Marks with Grade categories
        df_viz = st.session_state.df.copy()
        df_viz["Status"] = np.where(df_viz[TARGET_COLUMN] >= 75, "Distinction (≥75)",
                           np.where(df_viz[TARGET_COLUMN] >= 40, "Pass (40-74)", "Fail (<40)"))
        fig_att = px.box(
            df_viz,
            x="Status",
            y="attendance_percentage",
            color="Status",
            color_discrete_map={
                "Distinction (≥75)": "#10B981",
                "Pass (40-74)": "#6366F1",
                "Fail (<40)": "#EF4444"
            },
            title="Attendance Distribution across Academic Outcomes",
            labels={"attendance_percentage": "Attendance %", "Status": "Academic Outcome"}
        )
        fig_att.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_att, use_container_width=True)

    with chart_col4:
        # Correlation Heatmap
        corr = st.session_state.df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr().round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix: Academic Factors vs Exam Marks",
            labels=dict(color="Pearson Corr")
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# 2. PREDICT EXAM MARKS
# ---------------------------------------------------------
elif nav_selection == "🔮 Predict Exam Marks":
    st.markdown(
        render_hero(
            "🔮 Predict Student Examination Marks",
            "Fill in the academic details below to estimate expected exam marks, confidence intervals, grade category, and personalized academic feedback."
        ),
        unsafe_allow_html=True
    )

    col_form, col_pred = st.columns([1.1, 1.0], gap="large")

    with col_form:
        st.markdown("### 📝 **Enter Student Profile**")
        st.caption("Adjust sliders or enter specific numeric values:")

        # Quick preset buttons
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("💡 Topper Preset", use_container_width=True):
                st.session_state["p_hours"] = 7.5
                st.session_state["p_att"] = 92.0
                st.session_state["p_assign"] = 90.0
                st.session_state["p_prev"] = 88.0
                st.session_state["p_internal"] = 45.0
                st.session_state["p_days"] = 50
        with preset_cols[1]:
            if st.button("⚡ Average Preset", use_container_width=True):
                st.session_state["p_hours"] = 4.0
                st.session_state["p_att"] = 78.0
                st.session_state["p_assign"] = 72.0
                st.session_state["p_prev"] = 68.0
                st.session_state["p_internal"] = 35.0
                st.session_state["p_days"] = 30
        with preset_cols[2]:
            if st.button("⚠️ At-Risk Preset", use_container_width=True):
                st.session_state["p_hours"] = 1.5
                st.session_state["p_att"] = 55.0
                st.session_state["p_assign"] = 45.0
                st.session_state["p_prev"] = 42.0
                st.session_state["p_internal"] = 20.0
                st.session_state["p_days"] = 12

        hours = st.slider(
            "⏰ Daily Study Hours",
            min_value=0.5,
            max_value=14.0,
            value=st.session_state.get("p_hours", 4.5),
            step=0.5,
            help="Average daily uninterrupted study time",
            key="p_hours"
        )

        attendance = st.slider(
            "📊 Attendance Percentage (%)",
            min_value=40.0,
            max_value=100.0,
            value=st.session_state.get("p_att", 82.0),
            step=1.0,
            help="Percentage of total classes/lectures attended",
            key="p_att"
        )

        c1, c2 = st.columns(2)
        with c1:
            assignments = st.slider(
                "📋 Assignment Score (/ 100)",
                min_value=20.0,
                max_value=100.0,
                value=st.session_state.get("p_assign", 78.0),
                step=1.0,
                key="p_assign"
            )
            prev_exam = st.slider(
                "📜 Previous Exam Marks (/ 100)",
                min_value=20.0,
                max_value=100.0,
                value=st.session_state.get("p_prev", 72.0),
                step=1.0,
                key="p_prev"
            )
        with c2:
            internals = st.slider(
                "🧪 Internal Marks (/ 50)",
                min_value=10.0,
                max_value=50.0,
                value=st.session_state.get("p_internal", 38.0),
                step=1.0,
                key="p_internal"
            )
            study_days = st.slider(
                "🗓️ Study Days Before Exam",
                min_value=5,
                max_value=90,
                value=st.session_state.get("p_days", 35),
                step=1,
                key="p_days"
            )

        student_inputs = {
            "study_hours": float(hours),
            "attendance_percentage": float(attendance),
            "assignment_score": float(assignments),
            "previous_exam_marks": float(prev_exam),
            "internal_marks": float(internals),
            "study_days": int(study_days)
        }

    with col_pred:
        st.markdown(f"### 🎯 **Prediction Output ({st.session_state.selected_model_name})**")

        pred_res = predict_exam_marks(
            active_model_res["pipeline"],
            student_inputs,
            active_model_res["rmse"]
        )

        # Prominent Result Card
        st.markdown(render_prediction_badge(pred_res), unsafe_allow_html=True)

        # Plotly Gauge Indicator
        pred_val = pred_res["predicted_mark"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Exam Score Meter", 'font': {'size': 18}},
            number={'suffix': " / 100", 'font': {'size': 26, 'color': "#818CF8"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#6366F1"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [60, 75], 'color': 'rgba(59, 130, 246, 0.25)'},
                    {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#10B981", 'width': 4},
                    'thickness': 0.75,
                    'value': 75.0
                }
            }
        ))
        fig_gauge.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Metric breakdown comparing to class average
        avg_exam = st.session_state.df[TARGET_COLUMN].mean()
        diff = pred_val - avg_exam
        st.info(
            f"📊 **Benchmark:** This student is predicted to score **{abs(diff):.1f} marks "
            f"{'above' if diff >= 0 else 'below'}** the cohort average ({avg_exam:.1f}/100)."
        )

    # Recommendations Section
    st.markdown("---")
    st.markdown("### 💡 **Personalized Action Recommendations for This Student**")
    recommendations = generate_student_recommendations(student_inputs, pred_res)

    r_cols = st.columns(len(recommendations) if len(recommendations) <= 3 else 3)
    for idx, rec in enumerate(recommendations):
        col = r_cols[idx % len(r_cols)]
        with col:
            st.markdown(
                f"""
                <div class="rec-card {rec['type']}">
                    <span class="badge-tag">{rec['impact']}</span>
                    <div class="rec-title">{rec['icon']} {rec['title']}</div>
                    <div class="rec-body">{rec['message']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------------------------------------------
# 3. TARGET MARKS PLANNER (Extra Feature 1)
# ---------------------------------------------------------
elif nav_selection == "🎯 Target Marks Planner":
    st.markdown(
        render_hero(
            "🎯 Target Marks Planner & Goal Solver",
            "Want a specific score on the upcoming exam? Enter your desired target mark, and the solver calculates exactly how many daily study hours or attendance boost you need to reach it!"
        ),
        unsafe_allow_html=True
    )

    t_col1, t_col2 = st.columns([1, 1.2], gap="large")

    with t_col1:
        st.markdown("### 🏹 **Set Your Target Goal**")
        target_score = st.slider(
            "Desired Target Exam Mark (/ 100)",
            min_value=40.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

        st.markdown("#### **Your Current Baseline Profile**")
        curr_hours = st.slider("Current Study Hours", 0.5, 12.0, 4.0, 0.5, key="tg_hours")
        curr_att = st.slider("Current Attendance %", 40.0, 100.0, 78.0, 1.0, key="tg_att")
        curr_assign = st.slider("Current Assignment Score", 20.0, 100.0, 75.0, 1.0, key="tg_assign")
        curr_prev = st.slider("Previous Exam Marks", 20.0, 100.0, 70.0, 1.0, key="tg_prev")
        curr_internal = st.slider("Internal Marks", 10.0, 50.0, 35.0, 1.0, key="tg_internal")
        curr_days = st.slider("Study Days Remaining", 5, 90, 30, 1, key="tg_days")

        curr_inputs = {
            "study_hours": float(curr_hours),
            "attendance_percentage": float(curr_att),
            "assignment_score": float(curr_assign),
            "previous_exam_marks": float(curr_prev),
            "internal_marks": float(curr_internal),
            "study_days": int(curr_days)
        }

    with t_col2:
        st.markdown("### 🚀 **Goal Feasibility Analysis**")

        goal_res = simulate_target_goal(
            active_model_res["pipeline"],
            curr_inputs,
            target_score,
            active_model_res["rmse"]
        )

        current_pred = goal_res["current_pred"]
        gap = goal_res["gap"]

        # Comparison metrics
        gm1, gm2, gm3 = st.columns(3)
        with gm1:
            st.metric("Current Projection", f"{current_pred:.1f} / 100")
        with gm2:
            st.metric("Target Goal", f"{target_score:.1f} / 100")
        with gm3:
            st.metric("Score Gap", f"{gap:+.1f} Marks", delta=f"{gap:+.1f}", delta_color="inverse" if gap > 0 else "normal")

        st.progress(min(current_pred / max(target_score, 1.0), 1.0))

        if goal_res["status"] == "achieved":
            st.balloons()
            st.success(f"🎉 **Target Met!** Your current projection of **{current_pred:.1f}** already exceeds your target goal of **{target_score:.1f}**.")
        else:
            st.warning(f"⚠️ **Gap of {gap:.1f} marks to close.** Here are actionable ways to bridge this deficit:")

            path1, path2 = st.columns(2)
            with path1:
                st.markdown("#### ⏰ **Option A: Study Hours Boost**")
                if goal_res["needed_hours"] is not None:
                    add_hrs = round(goal_res["needed_hours"] - curr_hours, 1)
                    st.success(
                        f"""
                        - **Target Study Time:** `{goal_res['needed_hours']} hrs/day`
                        - **Increase Needed:** `+{add_hrs} hrs/day`
                        - **Feasibility:** {'High' if goal_res['needed_hours'] <= 8 else 'Challenging'}
                        """
                    )
                else:
                    st.error("Study hours alone cannot bridge this gap due to diminishing returns. Combine with attendance boost!")

            with path2:
                st.markdown("#### 📚 **Option B: Attendance Boost**")
                if goal_res["needed_attendance"] is not None:
                    add_att = round(goal_res["needed_attendance"] - curr_att, 1)
                    st.info(
                        f"""
                        - **Target Attendance:** `{goal_res['needed_attendance']}%`
                        - **Increase Needed:** `+{add_att}%`
                        - **Feasibility:** {'Achievable' if goal_res['needed_attendance'] <= 100 else 'Past Max'}
                        """
                    )
                else:
                    st.info("Attendance alone is not enough; daily study hours must also increase.")

        # Interactive Sensitivity Curve
        st.markdown("#### 📈 **Study Hours Sensitivity Curve for Your Profile**")
        h_range = np.linspace(1.0, 12.0, 23)
        curve_scores = []
        for h in h_range:
            sim_dict = curr_inputs.copy()
            sim_dict["study_hours"] = float(h)
            p = predict_exam_marks(active_model_res["pipeline"], sim_dict, active_model_res["rmse"])["predicted_mark"]
            curve_scores.append(p)

        curve_df = pd.DataFrame({"Daily Study Hours": h_range, "Projected Marks": curve_scores})
        fig_curve = px.line(
            curve_df,
            x="Daily Study Hours",
            y="Projected Marks",
            title="Estimated Exam Score vs Study Hours (Current Baseline)",
            markers=True
        )
        fig_curve.add_hline(y=target_score, line_dash="dash", line_color="#10B981", annotation_text=f"Target Goal ({target_score})")
        fig_curve.add_vline(x=curr_hours, line_dash="dot", line_color="#6366F1", annotation_text=f"Current ({curr_hours}h)")
        fig_curve.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_curve, use_container_width=True)

# ---------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
elif nav_selection == "⚙️ Model Training & Evaluation":
    st.markdown(
        render_hero(
            "⚙️ Model Training, Leaderboard & Metrics",
            "Train and evaluate multiple regression models on training/testing splits. Inspect MAE, MSE, RMSE, R² scores, actual-vs-predicted curves, and error residual distributions."
        ),
        unsafe_allow_html=True
    )

    # Configuration Controls
    with st.expander("🛠️ **Training Split & Generation Configuration**", expanded=False):
        c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
        with c_cfg1:
            split_pct = st.slider("Test Set Split Ratio (%)", 10, 40, int(st.session_state.test_size * 100), 5)
        with c_cfg2:
            seed_val = st.number_input("Random Seed (Reproducibility)", value=st.session_state.random_seed, step=1)
        with c_cfg3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Retrain All Models", use_container_width=True):
                st.session_state.test_size = split_pct / 100.0
                st.session_state.random_seed = int(seed_val)
                st.cache_resource.clear()
                st.rerun()

    # Model Leaderboard
    st.markdown("### 🏆 **Model Comparison Leaderboard**")
    st.caption("All models trained with Standard Scaling on identical train/test splits:")

    # Highlight best model
    st.dataframe(
        leaderboard_df.drop(columns=["_r2_val", "_rmse_val"]),
        use_container_width=True,
        hide_index=True
    )

    # Active Model Metrics Display
    st.markdown(f"### 📊 **Active Model Deep-Dive: `{st.session_state.selected_model_name}`**")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("R² Score (Accuracy)", f"{active_model_res['r2']:.4f}")
    with m2:
        st.metric("RMSE (Root Mean Sq Err)", f"{active_model_res['rmse']:.2f} marks")
    with m3:
        st.metric("MAE (Mean Abs Err)", f"{active_model_res['mae']:.2f} marks")
    with m4:
        st.metric("MSE (Mean Sq Err)", f"{active_model_res['mse']:.2f}")
    with m5:
        st.metric("Train R²", f"{active_model_res['train_r2']:.4f}")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        # Actual vs Predicted Scatter Plot
        eval_df = pd.DataFrame({
            "Actual Marks": active_model_res["y_test"],
            "Predicted Marks": active_model_res["y_test_pred"]
        })
        fig_eval = px.scatter(
            eval_df,
            x="Actual Marks",
            y="Predicted Marks",
            title=f"Actual vs Predicted Exam Marks ({st.session_state.selected_model_name})",
            labels={"Actual Marks": "Actual Exam Marks (/ 100)", "Predicted Marks": "Predicted Marks (/ 100)"},
            opacity=0.7,
            color_discrete_sequence=["#6366F1"]
        )
        # 45-degree perfect prediction line
        min_val = min(eval_df["Actual Marks"].min(), eval_df["Predicted Marks"].min())
        max_val = max(eval_df["Actual Marks"].max(), eval_df["Predicted Marks"].max())
        fig_eval.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Prediction (y = x)",
            line=dict(color="#EF4444", dash="dash")
        ))
        fig_eval.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_eval, use_container_width=True)

    with eval_col2:
        # Residuals Distribution Plot
        res_df = pd.DataFrame({"Residuals (Actual - Predicted)": active_model_res["residuals"]})
        fig_res = px.histogram(
            res_df,
            x="Residuals (Actual - Predicted)",
            nbins=30,
            title="Prediction Residuals Distribution (Errors)",
            color_discrete_sequence=["#10B981"],
            marginal="rug"
        )
        fig_res.add_vline(x=0, line_dash="solid", line_color="#EF4444", annotation_text="Zero Error")
        fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_res, use_container_width=True)

    # Models Comparison Bar Chart
    st.markdown("### 📊 **Comparative Benchmark across All Models**")
    fig_comp = px.bar(
        leaderboard_df,
        x="Model",
        y="_r2_val",
        color="_rmse_val",
        color_continuous_scale="Blues_r",
        title="R² Score Comparison (Higher is better, shaded by lower RMSE)",
        labels={"_r2_val": "R² Score", "_rmse_val": "RMSE", "Model": "Regression Model"},
        text="_r2_val"
    )
    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# 5. INSIGHTS & RECOMMENDATIONS
# ---------------------------------------------------------
elif nav_selection == "💡 Insights & Recommendations":
    st.markdown(
        render_hero(
            "💡 Academic Insights & Feature Impact",
            "Discover which academic behaviors have the strongest statistical impact on final exam marks. Formulate evidence-based study strategies to maximize performance."
        ),
        unsafe_allow_html=True
    )

    fi_df = get_feature_importance_df(active_model_res["pipeline"], FEATURE_COLUMNS)

    ins_col1, ins_col2 = st.columns([1.2, 1], gap="large")

    with ins_col1:
        st.markdown(f"### 📊 **Feature Importance ({st.session_state.selected_model_name})**")
        st.caption(f"Computation Method: {fi_df['Method'].iloc[0]}")

        fig_fi = px.bar(
            fi_df,
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            color="Importance_Pct",
            color_continuous_scale="Blues",
            title="Relative Factor Importance (%) on Final Exam Marks",
            labels={"Importance_Pct": "Importance (%)", "Feature": "Academic Metric"},
            text="Importance_Pct"
        )
        fig_fi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with ins_col2:
        st.markdown("### 🔍 **Key Academic Takeaways**")
        top_feature = fi_df.iloc[0]["Feature"]
        second_feature = fi_df.iloc[1]["Feature"]

        st.markdown(
            f"""
            1. **Primary Performance Driver:** **{top_feature}** accounts for **{fi_df.iloc[0]['Importance_Pct']}%** of model decision weight.
            2. **Secondary Pillar:** **{second_feature}** is the second most influential factor (**{fi_df.iloc[1]['Importance_Pct']}%**).
            3. **Consistency vs Cramming:** Daily study hours combined with continuous assessment (internal marks & assignments) outperform short-term cramming study days.
            4. **Attendance Multiplier:** Regular attendance (>80%) significantly reduces variance in exam performance by ensuring coverage of exam-relevant problems.
            """
        )

        st.markdown("### 📋 **General Recommendations for Students**")
        st.info("💡 **Rule of 5:** Studying 4.5 to 5.5 hours daily with regular 10-minute pauses maximizes retention while preventing diminishing returns.")
        st.success("✅ **Continuous Internal Focus:** Internal assessment and assignments represent safe foundational points that buffer difficult final exam questions.")
        st.warning("⚠️ **Attendance Baseline:** Maintain minimum 75% attendance to prevent academic penalties and stay updated on professors' test hints.")

    # Printable Student Diagnostic Report Generator (Extra Feature 2)
    st.markdown("---")
    st.markdown("### 📑 **Student Diagnostic Report Generator (Extra Feature)**")
    st.caption("Generate a personalized summary card and download report text for any student profile:")

    gen_col1, gen_col2 = st.columns([1, 1])
    with gen_col1:
        s_name = st.text_input("Student Name / Roll Number", "John Doe (STU1088)")
        s_hrs = st.number_input("Daily Study Hours", 0.5, 14.0, 5.0, 0.5)
        s_att = st.number_input("Attendance (%)", 40.0, 100.0, 85.0, 1.0)
        s_internal = st.number_input("Internal Score (/50)", 10.0, 50.0, 38.0, 1.0)

    with gen_col2:
        s_assign = st.number_input("Assignment Score (/100)", 20.0, 100.0, 80.0, 1.0)
        s_prev = st.number_input("Previous Exam Marks (/100)", 20.0, 100.0, 75.0, 1.0)
        s_days = st.number_input("Study Days Left", 5, 90, 30, 1)

    s_inputs = {
        "study_hours": s_hrs,
        "attendance_percentage": s_att,
        "assignment_score": s_assign,
        "previous_exam_marks": s_prev,
        "internal_marks": s_internal,
        "study_days": int(s_days)
    }
    s_pred = predict_exam_marks(active_model_res["pipeline"], s_inputs, active_model_res["rmse"])
    s_recs = generate_student_recommendations(s_inputs, s_pred)

    report_text = f"""============================================================
EXAM MARKS PREDICTION & ACADEMIC PRESCRIPTION REPORT
============================================================
Student Identifier : {s_name}
Predictive Engine  : {st.session_state.selected_model_name}
Expected Exam Mark : {s_pred['predicted_mark']} / 100
95% Conf. Interval : {s_pred['lower_bound_95']} - {s_pred['upper_bound_95']} Marks
Expected Grade     : {s_pred['grade_info']['grade']} ({s_pred['grade_info']['class']})

--- STUDENT PROFILE ---
• Daily Study Hours  : {s_hrs} hrs/day
• Attendance Rate    : {s_att}%
• Assignment Marks   : {s_assign} / 100
• Previous Exam Mark : {s_prev} / 100
• Internal Marks     : {s_internal} / 50
• Study Days Left    : {s_days} days

--- ACTION RECOMMENDATIONS ---
""" + "\n".join([f"• [{r['impact']}] {r['title']}: {r['message']}" for r in s_recs]) + "\n============================================================"

    st.download_button(
        label="📥 Download Student Performance Report (.txt)",
        data=report_text,
        file_name=f"student_report_{s_name.replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ---------------------------------------------------------
# 6. DATASET EXPLORER
# ---------------------------------------------------------
elif nav_selection == "📁 Dataset Explorer":
    st.markdown(
        render_hero(
            "📁 Dataset Explorer & Data Management",
            "Inspect raw student dataset records, review descriptive statistics, upload custom student datasets, or download the benchmark dataset."
        ),
        unsafe_allow_html=True
    )

    tab_view, tab_upload, tab_stats = st.tabs(["📋 View Data", "📤 Upload Custom CSV", "📊 Statistical Summary"])

    with tab_view:
        st.markdown(f"**Current Dataset:** `{len(st.session_state.df):,}` rows &bull; `{len(st.session_state.df.columns)}` columns")

        # Search / filter
        search_id = st.text_input("🔍 Search by Student ID (e.g. STU1005)", "")
        filtered_df = st.session_state.df
        if search_id:
            filtered_df = filtered_df[filtered_df["student_id"].str.contains(search_id, case=False, na=False)]

        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download CSV
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Current Dataset as CSV",
            data=csv_data,
            file_name="student_exam_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_upload:
        st.markdown("### 📤 **Upload Custom Student CSV**")
        st.caption("Upload your own CSV with student metrics. Required columns: " + ", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]))

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in user_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
                else:
                    clean_df, issues = clean_and_validate_dataframe(user_df)
                    st.session_state.df = clean_df
                    st.cache_resource.clear()
                    st.success(f"Successfully loaded and validated {len(clean_df)} records!")
                    if issues:
                        for issue in issues:
                            st.info(issue)
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing file: {e}")

        st.markdown("---")
        if st.button("🔄 Reset to Default Synthetic Benchmark Dataset", use_container_width=True):
            st.session_state.df = load_or_create_dataset(force_regenerate=True)
            st.cache_resource.clear()
            st.success("Reset to default synthetic dataset.")
            st.rerun()

    with tab_stats:
        st.markdown("### 📊 **Descriptive Statistics**")
        st.dataframe(st.session_state.df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T, use_container_width=True)
