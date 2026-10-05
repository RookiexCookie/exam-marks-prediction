"""
Custom CSS styles and UI component helpers for the Exam Marks Prediction App.
"""

CUSTOM_CSS = """
<style>
/* Modern Fonts and Clean Layout */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: #FFFFFF;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #FFFFFF, #E0E7FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #C7D2FE;
    font-weight: 400;
    max-width: 800px;
    line-height: 1.5;
    margin: 0;
}

/* KPI Metric Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #6366F1;
    box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.2);
}

.kpi-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4F46E5;
    margin: 0.3rem 0;
}

.kpi-label {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Prediction Highlight Card */
.score-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    color: #FFFFFF;
    border: 1px solid rgba(99, 102, 241, 0.4);
    box-shadow: 0 15px 30px -8px rgba(15, 23, 42, 0.5);
    margin-bottom: 1.5rem;
}

.score-number {
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.score-max {
    font-size: 1.5rem;
    font-weight: 600;
    color: #94A3B8;
}

.grade-pill {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-top: 0.8rem;
    letter-spacing: 0.02em;
}

.interval-box {
    margin-top: 1.2rem;
    padding: 0.8rem 1.2rem;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    font-size: 0.9rem;
    color: #CBD5E1;
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Recommendation Cards */
.rec-card {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.9rem;
    border-left: 4px solid #6366F1;
    border-top: 1px solid rgba(148, 163, 184, 0.15);
    border-right: 1px solid rgba(148, 163, 184, 0.15);
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
}

.rec-card.warning {
    border-left-color: #F59E0B;
}

.rec-card.danger {
    border-left-color: #EF4444;
}

.rec-card.success {
    border-left-color: #10B981;
}

.rec-card.info {
    border-left-color: #3B82F6;
}

.rec-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.rec-body {
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.45;
}

.badge-tag {
    float: right;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8;
    font-weight: 600;
}

/* Streamlit Button Styling */
div.stButton > button:first-child {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease-in-out;
}

/* Streamlit Tabs */
button[data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
}
</style>
"""

def render_hero(title: str, subtitle: str) -> str:
    return f"""
    <div class="hero-card">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """

def render_kpi(label: str, value: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val">{value}</div>
    </div>
    """

def render_prediction_badge(result: dict) -> str:
    mark = result["predicted_mark"]
    low95 = result["lower_bound_95"]
    high95 = result["upper_bound_95"]
    grade_info = result["grade_info"]

    return f"""
    <div class="score-card">
        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 0.5rem;">
            Estimated Examination Performance
        </div>
        <div class="score-number">
            {mark:.1f} <span class="score-max">/ 100</span>
        </div>
        <div>
            <span class="grade-pill" style="background: {grade_info['color']}22; color: {grade_info['color']}; border: 1px solid {grade_info['color']};">
                Grade {grade_info['grade']} &bull; {grade_info['class']}
            </span>
        </div>
        <div class="interval-box">
            🎯 <strong>95% Prediction Interval:</strong> {low95:.1f} &mdash; {high95:.1f} Marks (Margin of error: &plusmn;{(high95 - mark):.1f})
        </div>
    </div>
    """
