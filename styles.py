"""
Custom Modern Styling & Theme Configuration for Streamlit.
Provides responsive layout styling, glassmorphism cards, badge indicators,
and consistent aesthetic design tokens.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font and Base Styles */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Background & Main Container */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                #0b0f19;
    color: #e2e8f0;
}

/* Header Banner */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #06b6d4);
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 800px;
    line-height: 1.6;
    margin: 0;
}

/* Pill Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-ai {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-rose {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Metric KPI Card */
.metric-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.75));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 14px 30px -5px rgba(99, 102, 241, 0.18);
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.metric-sub {
    font-size: 0.82rem;
    color: #64748b;
}

/* Glassmorphism Section Container */
.glass-container {
    background: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

/* Section Headings */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 1.3rem;
}

/* Recommendation Box */
.rec-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.05));
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    color: #e2e8f0;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Prediction Result Highlight Card */
.prediction-score-display {
    text-align: center;
    padding: 2.2rem 1.5rem;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

.score-number {
    font-size: 4.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0.5rem 0;
}

/* Risk Color Modifiers */
.score-green {
    color: #34d399;
    text-shadow: 0 0 25px rgba(52, 211, 153, 0.35);
}

.score-amber {
    color: #fbbf24;
    text-shadow: 0 0 25px rgba(251, 191, 36, 0.35);
}

.score-red {
    color: #f87171;
    text-shadow: 0 0 25px rgba(248, 113, 113, 0.35);
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0c111d;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Streamlit Native Tabs Enhancement */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
}

/* Custom Buttons */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    font-size: 0.95rem;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    transition: all 0.2s ease;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
"""

def setup_matplotlib_theme():
    """Configure Matplotlib to match the sleek dark UI design system."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': '#111827',
        'axes.facecolor': '#111827',
        'axes.edgecolor': '#374151',
        'axes.labelcolor': '#9CA3AF',
        'xtick.color': '#9CA3AF',
        'ytick.color': '#9CA3AF',
        'grid.color': '#1F2937',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'text.color': '#F3F4F6',
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.autolayout': True
    })
