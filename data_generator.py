"""
Data Generator for AI-Based Student Performance Prediction System.
Generates realistic student academic, behavioral, and demographic data
with scientifically grounded correlations and non-linear patterns.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset simulating real-world student academic performance.
    """
    np.random.seed(random_state)
    
    # 1. Identification
    student_ids = [f"STU-{1000 + i}" for i in range(n_samples)]
    
    # 2. Demographic & Background
    parent_edu_options = ["High School", "Associate Degree", "Bachelor's Degree", "Master's / PhD"]
    parent_edu_probs = [0.25, 0.25, 0.35, 0.15]
    parental_education = np.random.choice(parent_edu_options, size=n_samples, p=parent_edu_probs)
    
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])
    tutoring = np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62])
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    
    study_env_options = ["Quiet / Dedicated", "Moderate Noise", "Noisy / Distracting"]
    study_env = np.random.choice(study_env_options, size=n_samples, p=[0.52, 0.33, 0.15])
    
    stress_level_options = ["Low", "Medium", "High"]
    stress_level = np.random.choice(stress_level_options, size=n_samples, p=[0.28, 0.48, 0.24])

    # 3. Behavioral & Academic Features
    # Study hours (skewed towards 8-18 hrs/week)
    study_hours = np.clip(np.random.gamma(shape=3.5, scale=4.0, size=n_samples), 2.0, 38.0).round(1)
    
    # Attendance Rate (mostly high, but long left tail for at-risk)
    raw_attendance = np.random.beta(a=7, b=2, size=n_samples) * 100
    attendance_rate = np.clip(raw_attendance, 40.0, 100.0).round(1)
    
    # Previous Exam Score (approx bell curve centered at 68)
    prev_scores = np.random.normal(loc=68, scale=14, size=n_samples)
    prev_scores = np.clip(prev_scores, 25.0, 99.0).round(1)
    
    # Sleep hours (normal centered at 7.1, with slight variations)
    sleep_hours = np.clip(np.random.normal(loc=7.1, scale=1.1, size=n_samples), 4.0, 10.0).round(1)
    
    # Assignments completed rate (correlated with study hours & attendance)
    assign_base = (0.5 * (attendance_rate / 100) + 0.3 * (study_hours / 35) + 0.2 * np.random.rand(n_samples)) * 100
    assignments_rate = np.clip(assign_base + np.random.normal(0, 5, n_samples), 30.0, 100.0).round(1)

    # 4. Synthesize Final Exam Score using non-linear educational dynamics
    # Normalized features (0 to 1 scale)
    norm_prev = (prev_scores - 25) / (99 - 25)
    norm_study = np.minimum(study_hours / 28.0, 1.2)  # diminishing return after 28 hrs
    norm_attend = (attendance_rate - 40) / 60.0
    norm_assign = assignments_rate / 100.0
    
    # Sleep penalty: ideal is 7-8 hours; <5 or >9 has cognitive dip
    sleep_efficiency = 1.0 - 0.25 * ((sleep_hours - 7.5) / 3.0) ** 2
    sleep_efficiency = np.clip(sleep_efficiency, 0.65, 1.0)
    
    # Categorical boosts / penalties
    edu_boost = {
        "High School": -1.5,
        "Associate Degree": 0.5,
        "Bachelor's Degree": 2.0,
        "Master's / PhD": 3.5
    }
    edu_effect = np.array([edu_boost[e] for e in parental_education])
    
    stress_penalty = {
        "Low": 1.5,
        "Medium": 0.0,
        "High": -3.5
    }
    stress_effect = np.array([stress_penalty[s] for s in stress_level])
    
    env_bonus = {
        "Quiet / Dedicated": 2.0,
        "Moderate Noise": 0.0,
        "Noisy / Distracting": -2.5
    }
    env_effect = np.array([env_bonus[v] for v in study_env])
    
    tutoring_effect = np.where(tutoring == "Yes", 3.2, 0.0)
    internet_effect = np.where(internet_access == "Yes", 1.8, -2.5)
    extra_effect = np.where(extracurricular == "Yes", 1.0, 0.0)
    
    # Interaction synergy (study hours + attendance synergize)
    synergy = (norm_study * norm_attend) * 5.0
    
    # Weighted score synthesis (out of 100)
    base_score = (
        0.35 * (norm_prev * 100) +
        0.22 * (norm_study * 75) +
        0.20 * (norm_attend * 85) +
        0.15 * (norm_assign * 80) +
        synergy +
        edu_effect +
        stress_effect +
        env_effect +
        tutoring_effect +
        internet_effect +
        extra_effect
    ) * sleep_efficiency
    
    # Natural Gaussian noise representing exam-day factors
    exam_day_noise = np.random.normal(loc=0.0, scale=3.8, size=n_samples)
    final_score = np.clip(base_score + exam_day_noise, 15.0, 99.5).round(1)
    
    # 5. Risk Category & Classification labels
    # Academic categories
    conditions_cat = [
        final_score >= 80.0,
        (final_score >= 65.0) & (final_score < 80.0),
        (final_score >= 50.0) & (final_score < 65.0),
        final_score < 50.0
    ]
    choices_cat = ["Excellent (A)", "Good (B)", "Average (C)", "At-Risk (Fail)"]
    performance_category = np.select(conditions_cat, choices_cat, default="Average (C)")
    
    # Risk Level
    conditions_risk = [
        final_score < 50.0,
        (final_score >= 50.0) & (final_score < 65.0),
        final_score >= 65.0
    ]
    choices_risk = ["High Risk", "Moderate Risk", "Low Risk"]
    risk_level = np.select(conditions_risk, choices_risk, default="Moderate Risk")
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Rate": attendance_rate,
        "Past_Exam_Score": prev_scores,
        "Sleep_Hours": sleep_hours,
        "Assignments_Completion_Rate": assignments_rate,
        "Parental_Education": parental_education,
        "Internet_Access": internet_access,
        "Tutoring": tutoring,
        "Extracurricular_Activities": extracurricular,
        "Study_Environment": study_env,
        "Stress_Level": stress_level,
        "Final_Exam_Score": final_score,
        "Performance_Category": performance_category,
        "Risk_Level": risk_level
    })
    
    return df

def save_default_datasets(output_dir: str = "data"):
    """Generate and persist default train/test dataset and sample batch roster."""
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, "students_data.csv")
    batch_path = os.path.join(output_dir, "sample_batch_roster.csv")
    
    # Main training dataset
    df_main = generate_student_dataset(n_samples=1500, random_state=42)
    df_main.to_csv(master_path, index=False)
    print(f"Saved master dataset: {master_path} ({len(df_main)} records)")
    
    # Batch roster (unlabeled, simulating upcoming students to evaluate)
    df_batch = generate_student_dataset(n_samples=40, random_state=101)
    df_batch_unlabeled = df_batch.drop(columns=["Final_Exam_Score", "Performance_Category", "Risk_Level"])
    df_batch_unlabeled.to_csv(batch_path, index=False)
    print(f"Saved sample batch roster: {batch_path} ({len(df_batch_unlabeled)} records)")
    
    return master_path, batch_path

if __name__ == "__main__":
    save_default_datasets("data")
