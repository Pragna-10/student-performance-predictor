"""
Machine Learning Pipelines, Training, Evaluation, and Explainability Utilities
for Student Performance Prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Feature grouping constants
NUMERIC_FEATURES = [
    "Study_Hours_Per_Week",
    "Attendance_Rate",
    "Past_Exam_Score",
    "Sleep_Hours",
    "Assignments_Completion_Rate"
]

CATEGORICAL_FEATURES = [
    "Parental_Education",
    "Internet_Access",
    "Tutoring",
    "Extracurricular_Activities",
    "Study_Environment",
    "Stress_Level"
]

TARGET_REGRESSION = "Final_Exam_Score"
TARGET_CLASSIFICATION = "Performance_Category"

def create_preprocessor():
    """Build sklearn ColumnTransformer for numeric and categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

def get_regression_models(rf_estimators: int = 100, rf_depth: int = 12, gb_estimators: int = 100, learning_rate: float = 0.1) -> Dict[str, Any]:
    """Return dictionary of available regression model estimators."""
    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=gb_estimators,
            learning_rate=learning_rate,
            random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }

def get_classification_models(rf_estimators: int = 100, rf_depth: int = 10) -> Dict[str, Any]:
    """Return dictionary of available classification model estimators."""
    return {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            random_state=42
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=rf_estimators,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

def train_and_evaluate_regression(
    df: pd.DataFrame,
    model_name: str,
    test_size: float = 0.2,
    rf_estimators: int = 100,
    rf_depth: int = 12,
    gb_estimators: int = 100,
    learning_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Train a regression pipeline and calculate comprehensive metrics.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_REGRESSION]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    models = get_regression_models(
        rf_estimators=rf_estimators,
        rf_depth=rf_depth,
        gb_estimators=gb_estimators,
        learning_rate=learning_rate
    )
    estimator = models[model_name]
    
    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("model", estimator)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
    
    # Extract feature importance if available
    feature_importance_df = extract_feature_importance(pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    
    return {
        "pipeline": pipeline,
        "model_name": model_name,
        "metrics": {
            "R2 Score": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "MAPE (%)": round(float(mape), 2)
        },
        "y_test": y_test.values,
        "y_pred": y_pred,
        "residuals": (y_test - y_pred).values,
        "feature_importance": feature_importance_df,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

def train_and_evaluate_classification(
    df: pd.DataFrame,
    model_name: str,
    test_size: float = 0.2,
    rf_estimators: int = 100,
    rf_depth: int = 10
) -> Dict[str, Any]:
    """
    Train a classification pipeline and calculate accuracy, F1, precision, recall, and confusion matrix.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_CLASSIFICATION]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models = get_classification_models(rf_estimators=rf_estimators, rf_depth=rf_depth)
    estimator = models[model_name]
    
    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("model", estimator)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    classes = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    return {
        "pipeline": pipeline,
        "model_name": model_name,
        "metrics": {
            "Accuracy": round(float(acc) * 100, 2),
            "F1-Score": round(float(f1) * 100, 2),
            "Precision": round(float(prec) * 100, 2),
            "Recall": round(float(rec) * 100, 2)
        },
        "classes": classes,
        "confusion_matrix": cm,
        "y_test": y_test.values,
        "y_pred": y_pred
    }

def extract_feature_importance(pipeline: Pipeline, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Extract and format feature importances or coefficients from the fitted pipeline."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        
        # Get one-hot feature names
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        all_feature_names = num_cols + encoded_cat_names
        
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
                
        if importances is not None and len(importances) == len(all_feature_names):
            df_imp = pd.DataFrame({
                "Feature": all_feature_names,
                "Importance": importances
            })
            # Clean up display names
            df_imp["Clean_Feature"] = df_imp["Feature"].apply(
                lambda x: x.replace("_", " ").title()
            )
            df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
            # Normalize to 100%
            total = df_imp["Importance"].sum()
            if total > 0:
                df_imp["Weight (%)"] = ((df_imp["Importance"] / total) * 100).round(2)
            else:
                df_imp["Weight (%)"] = 0.0
            return df_imp
    except Exception as e:
        pass
    
    return pd.DataFrame()

def predict_single_student(pipeline: Pipeline, student_dict: Dict[str, Any]) -> Tuple[float, str, str]:
    """
    Run prediction for a single student dictionary input.
    Returns: (predicted_score, performance_category, risk_level)
    """
    input_df = pd.DataFrame([student_dict])
    pred_score = float(pipeline.predict(input_df)[0])
    pred_score = np.clip(pred_score, 0.0, 100.0).round(1)
    
    # Map to categories
    if pred_score >= 80.0:
        cat = "Excellent (A)"
        risk = "Low Risk"
    elif pred_score >= 65.0:
        cat = "Good (B)"
        risk = "Low Risk"
    elif pred_score >= 50.0:
        cat = "Average (C)"
        risk = "Moderate Risk"
    else:
        cat = "At-Risk (Fail)"
        risk = "High Risk"
        
    return pred_score, cat, risk

def generate_actionable_insights(student_dict: Dict[str, Any], predicted_score: float, pipeline: Pipeline) -> List[Dict[str, Any]]:
    """
    Analyze student input parameters, simulate improvements, and generate actionable recommendations.
    """
    recommendations = []
    
    # 1. Study Hours check
    study_hrs = student_dict["Study_Hours_Per_Week"]
    if study_hrs < 15.0:
        boosted_dict = student_dict.copy()
        target_study = min(study_hrs + 5.0, 25.0)
        boosted_dict["Study_Hours_Per_Week"] = target_study
        new_score = float(pipeline.predict(pd.DataFrame([boosted_dict]))[0])
        diff = round(new_score - predicted_score, 1)
        if diff > 0.5:
            recommendations.append({
                "category": "Study Habits",
                "icon": "📚",
                "action": f"Increase weekly study time from {study_hrs}h to {target_study}h (+5 hrs/week).",
                "impact": f"+{diff} points estimated gain",
                "severity": "high" if predicted_score < 60 else "medium"
            })
            
    # 2. Attendance Check
    attendance = student_dict["Attendance_Rate"]
    if attendance < 85.0:
        boosted_dict = student_dict.copy()
        target_attend = min(attendance + 15.0, 98.0)
        boosted_dict["Attendance_Rate"] = target_attend
        new_score = float(pipeline.predict(pd.DataFrame([boosted_dict]))[0])
        diff = round(new_score - predicted_score, 1)
        if diff > 0.5:
            recommendations.append({
                "category": "Class Engagement",
                "icon": "🏫",
                "action": f"Improve class attendance rate from {attendance:.1f}% to {target_attend:.1f}%.",
                "impact": f"+{diff} points estimated gain",
                "severity": "high" if attendance < 75 else "medium"
            })

    # 3. Assignment Completion
    assign = student_dict["Assignments_Completion_Rate"]
    if assign < 85.0:
        boosted_dict = student_dict.copy()
        target_assign = min(assign + 20.0, 100.0)
        boosted_dict["Assignments_Completion_Rate"] = target_assign
        new_score = float(pipeline.predict(pd.DataFrame([boosted_dict]))[0])
        diff = round(new_score - predicted_score, 1)
        if diff > 0.5:
            recommendations.append({
                "category": "Coursework",
                "icon": "📝",
                "action": f"Complete all homework and submitted coursework (boost to {target_assign:.0f}%).",
                "impact": f"+{diff} points estimated gain",
                "severity": "high" if assign < 70 else "medium"
            })

    # 4. Tutoring
    if student_dict["Tutoring"] == "No" and predicted_score < 70:
        boosted_dict = student_dict.copy()
        boosted_dict["Tutoring"] = "Yes"
        new_score = float(pipeline.predict(pd.DataFrame([boosted_dict]))[0])
        diff = round(new_score - predicted_score, 1)
        if diff > 0.5:
            recommendations.append({
                "category": "Academic Support",
                "icon": "🤝",
                "action": "Enroll in peer tutoring or supplemental instruction sessions.",
                "impact": f"+{diff} points estimated gain",
                "severity": "high"
            })

    # 5. Sleep & Well-being
    sleep = student_dict["Sleep_Hours"]
    if sleep < 6.5:
        recommendations.append({
            "category": "Sleep & Health",
            "icon": "😴",
            "action": f"Prioritize regular sleep schedule (currently {sleep}h, aim for 7.5 - 8h).",
            "impact": "Crucial for memory consolidation & exam endurance",
            "severity": "medium"
        })
    elif sleep > 9.5:
        recommendations.append({
            "category": "Daily Routine",
            "icon": "⏰",
            "action": f"Structure daily schedule to optimize active study hours (currently sleeping {sleep}h).",
            "impact": "Improves study routine balance",
            "severity": "low"
        })
        
    # 6. Stress Management
    if student_dict["Stress_Level"] == "High":
        recommendations.append({
            "category": "Mental Wellness",
            "icon": "🧘",
            "action": "Utilize student counseling, test-anxiety workshops, or mindfulness routines.",
            "impact": "Reduces exam-day cognitive freeze",
            "severity": "high" if predicted_score < 60 else "medium"
        })

    # If student is already stellar
    if not recommendations:
        recommendations.append({
            "category": "Excellence Maintenance",
            "icon": "🌟",
            "action": "Outstanding academic habits! Maintain regular review intervals and consider peer mentoring.",
            "impact": "Maintains top tier standing",
            "severity": "low"
        })
        
    return recommendations

def calculate_sensitivity_curve(
    pipeline: Pipeline,
    base_student: Dict[str, Any],
    feature_name: str,
    feature_range: np.ndarray
) -> pd.DataFrame:
    """
    Vary a single numeric feature across its range while holding all others constant
    to plot its non-linear sensitivity curve.
    """
    records = []
    for val in feature_range:
        current = base_student.copy()
        current[feature_name] = val
        records.append(current)
        
    test_df = pd.DataFrame(records)
    preds = pipeline.predict(test_df)
    preds = np.clip(preds, 0.0, 100.0).round(1)
    
    return pd.DataFrame({
        feature_name: feature_range,
        "Predicted_Score": preds
    })
