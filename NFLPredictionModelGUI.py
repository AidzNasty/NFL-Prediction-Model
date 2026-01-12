#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Prediction Model - COMPLETE VERSION WITH ALL PAGES
Web App Version with Playoff Support - Fixed for NaN errors
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# PLAYOFF ROUND MAPPING
PLAYOFF_ROUNDS = {
    19: 'WildCard',
    20: 'Divisional',
    21: 'Conference',
    22: 'SuperBowl'
}

ROUND_TO_WEEK = {
    'WildCard': 19,
    'Divisional': 20,
    'Conference': 21,
    'SuperBowl': 22
}

def week_display_format(week_value):
    """Convert week number to display string - RENAMED to avoid conflict"""
    if pd.isna(week_value):
        return "Unknown"
    if isinstance(week_value, str):
        if week_value in ROUND_TO_WEEK:
            return week_value.replace('SuperBowl', 'Super Bowl')
        return week_value
    try:
        week_int = int(week_value)
        if week_int <= 18:
            return f"Week {week_int}"
        return PLAYOFF_ROUNDS.get(week_int, f"Week {week_int}")
    except:
        return str(week_value)

def normalize_week(week_value):
    """Convert any week format to consistent number"""
    if pd.isna(week_value):
        return None
    if isinstance(week_value, (int, float)):
        try:
            if np.isnan(week_value):
                return None
        except:
            pass
        return int(week_value)
    if isinstance(week_value, str):
        week_str = week_value.strip().lower()
        # Check exact matches first
        for key, val in ROUND_TO_WEEK.items():
            if week_str == key.lower():
                return val
        # Handle variations
        if 'wild' in week_str:
            return 19
        if 'division' in week_str:
            return 20
        if 'conference' in week_str or 'conf' in week_str:
            return 21
        if 'super' in week_str or 'bowl' in week_str:
            return 22
        if week_str.startswith('week'):
            try:
                return int(week_str.split()[1])
            except:
                pass
    try:
        return int(week_value)
    except:
        return None

def safe_int(value, default=0):
    """Safely convert to int, handling NaN"""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except:
        return default

def safe_float(value, default=0.0):
    """Safely convert to float, handling NaN"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default

# Page config
st.set_page_config(
    page_title="NFL Prediction Model 2025-26",
    page_icon="🏈",
    layout="wide"
)

# Custom CSS - Modern, Neutral Design
st.markdown("""
    <style>
    /* Main App Styling - Neutral Colors */
    .main {
        background: #f5f5f5;
        padding: 2rem 1rem;
    }
    .stApp {
        background: #f5f5f5;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #000000 !important;
        text-align: center;
    }
    h1 {
        margin-bottom: 0.5rem;
    }
    
    /* Centered Container - Wider for Desktop */
    .centered-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Game Card - Neutral Design */
    .game-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border-color: #d0d0d0;
    }
    
    /* Winner Styling - Good Green */
    .winner {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem;
        background: #e8f5e9;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #4caf50;
    }
    .ml-winner {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem;
        background: #e8f5e9;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #4caf50;
    }
    
    /* Metric Card - Neutral */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: #fafafa;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #9e9e9e;
        margin: 1rem 0;
    }
    
    /* Team Display */
    .team-display {
        text-align: center;
        padding: 1rem;
    }
    
    /* Score Display */
    .score-display {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        background: #f0f0f0;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Probability Badge */
    .prob-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Differential Badge */
    .diff-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .game-card {
            padding: 1.5rem;
        }
        h1 {
            font-size: 1.75rem;
        }
        h2 {
            font-size: 1.5rem;
        }
        .centered-container {
            max-width: 100%;
            padding: 0 1rem;
        }
    }
    
    /* Spacing Improvements */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Sidebar Improvements - Neutral */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Streamlit default text color */
    .stMarkdown p, .stMarkdown li {
        color: #000000;
    }
    
    /* All text elements black */
    body, p, span, div, label, caption, .stText, .stMarkdown {
        color: #000000 !important;
    }
    
    /* Captions and labels */
    .stCaption, label {
        color: #000000 !important;
    }
    
    /* Better table styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NFL 2025-26 Prediction Model.xlsx'

def get_probability_color(probability):
    """Convert probability to color gradient"""
    try:
        prob = float(probability)
    except:
        prob = 0.5
    
    normalized = (prob - 0.5) / 0.5
    normalized = max(0, min(1, normalized))
    
    if normalized < 0.5:
        t = normalized * 2
        r = int(211 + (251 - 211) * t)
        g = int(47 + (192 - 47) * t)
        b = int(47 + (45 - 47) * t)
    else:
        t = (normalized - 0.5) * 2
        r = int(251 + (56 - 251) * t)
        g = int(192 + (142 - 192) * t)
        b = int(45 + (60 - 45) * t)
    
    color = f"#{r:02x}{g:02x}{b:02x}"
    bg_color = f"rgba({r}, {g}, {b}, 0.15)"
    return color, bg_color

@st.cache_data
def load_data():
    """Load data from Excel file with proper NaN handling"""
    try:
        # Load predictions
        df_raw = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=None)
        home_edge = safe_float(df_raw.iloc[0, 1], 1.916137)  # HomeEdge from B1
        
        df_raw_with_header = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=0)
        predictions = df_raw_with_header.iloc[:, 3:].reset_index(drop=True)
        
        # Handle Date column safely
        try:
            predictions['Date'] = pd.to_datetime(predictions['Date'], errors='coerce')
        except:
            predictions['Date'] = pd.NaT
        
        # Normalize week values
        predictions['Week_Num'] = predictions['Week'].apply(normalize_week)
        predictions['Week_Display'] = predictions['Week_Num'].apply(week_display_format)
        
        # Remove rows with no week number
        predictions = predictions[predictions['Week_Num'].notna()].copy()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Fill NaN values in standings with defaults
        standings = standings.fillna({
            'W': 0,
            'L': 0,
            'HomeWin%': 0.5,
            'AwayWin%': 0.5,
            'PTS': 0
        })
        
        # Try to load ML predictions if available
        try:
            ml_predictions = pd.read_csv('nfl_predictions.csv')
            ml_predictions['date'] = pd.to_datetime(ml_predictions['date'], errors='coerce')
            return predictions, standings, home_edge, ml_predictions
        except:
            return predictions, standings, home_edge, None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

def display_game(game, standings, home_edge, ml_pred=None):
    """Display a single game with predictions"""
    home_team = game['Home Team']
    away_team = game['Away Team']
    
    # Get team stats safely
    try:
        home_stats = standings[standings['Team'] == home_team].iloc[0]
        away_stats = standings[standings['Team'] == away_team].iloc[0]
    except:
        st.warning(f"Could not find team stats for {home_team} or {away_team}")
        return
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header - Week and Date
        col_header1, col_header2 = st.columns([1, 1])
        with col_header1:
            st.markdown(f"**{game['Week_Display']}**")
        with col_header2:
            try:
                date_str = game['Date'].strftime('%a, %b %d')
            except:
                date_str = "TBD"
            st.markdown(f"**{date_str}**")
        
        st.markdown("---")
        
        # Teams and Records
        col_away, col_vs, col_home = st.columns([2, 1, 2])
        
        with col_away:
            st.markdown(f"### 🏈 {away_team}")
            away_wins = safe_int(away_stats.get('W', 0))
            away_losses = safe_int(away_stats.get('L', 0))
            away_win_pct = safe_float(away_stats.get('AwayWin%', 0.5))
            st.caption(f"Record: {away_wins}-{away_losses}")
            st.caption(f"Away: {away_win_pct:.1%}")
        
        with col_vs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>@</h4>", unsafe_allow_html=True)
        
        with col_home:
            st.markdown(f"### 🏈 {home_team}")
            home_wins = safe_int(home_stats.get('W', 0))
            home_losses = safe_int(home_stats.get('L', 0))
            home_win_pct = safe_float(home_stats.get('HomeWin%', 0.5))
            st.caption(f"Record: {home_wins}-{home_losses}")
            st.caption(f"Home: {home_win_pct:.1%}")
        
        st.markdown("---")
        
        # Excel Prediction
        st.markdown("### 📊 Excel Model")
        
        excel_winner = game.get('Predictied Winner', 'TBD')
        excel_prob = safe_float(game.get('Strength of Win', 0.5))
        homefield = safe_float(game.get('HomeEdge Differential', 0))
        
        prob_color, prob_bg = get_probability_color(excel_prob)
        
        col_pred1, col_pred2 = st.columns([2, 1])
        
        with col_pred1:
            st.markdown(
                f'<div class="winner">{excel_winner}</div>',
                unsafe_allow_html=True
            )
        
        with col_pred2:
            st.markdown(
                f'<span class="prob-badge" style="background: {prob_bg}; color: {prob_color};">'
                f'{excel_prob:.1%}</span>',
                unsafe_allow_html=True
            )
            st.caption(f"HomeField: {homefield:+.2f}")
        
        # ML Prediction if available
        if ml_pred is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🤖 ML Model")
            
            ml_winner = ml_pred.get('ml_winner', 'TBD')
            ml_home_score = safe_int(ml_pred.get('ml_home_score', 0))
            ml_away_score = safe_int(ml_pred.get('ml_away_score', 0))
            
            # Parse confidence
            try:
                ml_conf_str = str(ml_pred.get('ml_confidence', '50%')).replace('%', '')
                ml_conf = float(ml_conf_str) / 100
            except:
                ml_conf = 0.5
            
            ml_color, ml_bg = get_probability_color(ml_conf)
            
            col_ml1, col_ml2 = st.columns([2, 1])
            
            with col_ml1:
                st.markdown(
                    f'<div class="ml-winner">{ml_winner}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="score-display">{away_team}: {ml_away_score} | {home_team}: {ml_home_score}</div>',
                    unsafe_allow_html=True
                )
            
            with col_ml2:
                st.markdown(
                    f'<span class="prob-badge" style="background: {ml_bg}; color: {ml_color};">'
                    f'{ml_conf:.1%}</span>',
                    unsafe_allow_html=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load data
    predictions, standings, home_edge, ml_predictions = load_data()
    
    if predictions is None:
        st.error("Failed to load data. Please check the Excel file.")
        st.info("Common issues:\n- Excel file not in same folder\n- Sheet names don't match\n- Empty or NaN values in critical columns")
        return
    
    # Sidebar
    st.sidebar.title("🏈 NFL Predictions")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Today's Games", "All Games", "Performance", "Model Comparison"]
    )
    
    # Main content
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    
    # Title
    st.title("🏈 NFL Prediction Model 2025-26")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TODAY'S GAMES PAGE
    if page == "Today's Games":
        st.subheader("🎯 Today's Games")
        st.markdown("<br>", unsafe_allow_html=True)
        
        today = pd.Timestamp.now().normalize()
        todays_games = predictions[predictions['Date'].dt.normalize() == today]
        
        if len(todays_games) == 0:
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info(f"No games scheduled for {today.strftime('%A, %B %d, %Y')}")
        else:
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.success(f"📅 {len(todays_games)} game(s) today")
            
            for idx, game in todays_games.iterrows():
                # Find ML prediction if exists
                ml_pred = None
                if ml_predictions is not None:
                    try:
                        ml_match = ml_predictions[
                            (ml_predictions['home_team'] == game['Home Team']) &
                            (ml_predictions['away_team'] == game['Away Team']) &
                            (ml_predictions['date'].dt.date == game['Date'].date())
                        ]
                        if len(ml_match) > 0:
                            ml_pred = ml_match.iloc[0]
                    except:
                        pass
                
                display_game(game, standings, home_edge, ml_pred)
    
    # ALL GAMES PAGE
    elif page == "All Games":
        st.subheader("📋 All Games")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Week filter
        unique_displays = sorted(predictions['Week_Display'].unique(), 
                                 key=lambda x: predictions[predictions['Week_Display']==x]['Week_Num'].iloc[0])
        selected_week = st.sidebar.selectbox("Filter by Week", ["All Weeks"] + list(unique_displays))
        
        if selected_week == "All Weeks":
            filtered_games = predictions
        else:
            filtered_games = predictions[predictions['Week_Display'] == selected_week]
        
        col_info, _, _ = st.columns([2, 1, 2])
        with col_info:
            st.info(f"Showing {len(filtered_games)} games")
        
        for idx, game in filtered_games.iterrows():
            # Find ML prediction if exists
            ml_pred = None
            if ml_predictions is not None:
                try:
                    ml_match = ml_predictions[
                        (ml_predictions['home_team'] == game['Home Team']) &
                        (ml_predictions['away_team'] == game['Away Team']) &
                        (ml_predictions['date'].dt.date == game['Date'].date())
                    ]
                    if len(ml_match) > 0:
                        ml_pred = ml_match.iloc[0]
                except:
                    pass
            
            display_game(game, standings, home_edge, ml_pred)
    
    # PERFORMANCE PAGE
    elif page == "Performance":
        st.subheader("📊 Model Performance")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Excel Model Performance
        correct_col = 'Locked Correct'
        completed = predictions[predictions[correct_col].isin(['YES', 'NO'])].copy()
        
        if len(completed) == 0:
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info("No completed games yet")
        else:
            completed = completed.sort_values('Date')
            
            # Overall stats
            total = len(completed)
            correct = (completed[correct_col] == 'YES').sum()
            accuracy = (correct / total * 100)
            
            col_metrics, _, _ = st.columns([1, 0.3, 1])
            with col_metrics:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📈 Excel Model - Overall Results")
                
                col1, col2, col3, col4 = st.columns(4, gap="small")
                
                with col1:
                    st.metric("Total Games", total)
                
                with col2:
                    st.metric("Correct", correct)
                
                with col3:
                    st.metric("Wrong", total - correct)
                
                with col4:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Week-by-week (using display names)
            st.markdown("<br>", unsafe_allow_html=True)
            week_stats = completed.groupby('Week_Display').agg({
                correct_col: [
                    ('correct', lambda x: (x == 'YES').sum()),
                    ('total', 'count')
                ]
            }).reset_index()
            week_stats.columns = ['Week', 'Correct', 'Total']
            week_stats['Accuracy'] = (week_stats['Correct'] / week_stats['Total'] * 100).round(1)
            
            # Sort by week number
            week_stats = week_stats.merge(
                completed[['Week_Display', 'Week_Num']].drop_duplicates(),
                left_on='Week',
                right_on='Week_Display'
            ).sort_values('Week_Num').drop(columns=['Week_Display', 'Week_Num'])
            
            col_week, _, _ = st.columns([1, 0.3, 1])
            with col_week:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Week-by-Week Performance")
                st.dataframe(week_stats, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # ML Model Performance
        if ml_predictions is not None and len(ml_predictions) > 0:
            st.markdown("<br>")
            st.markdown("---")
            st.markdown("### 🤖 ML Model Performance")
            
            ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
            
            if len(ml_completed) > 0:
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 'YES').sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                
                col_ml_metrics, _, _ = st.columns([1, 0.3, 1])
                with col_ml_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Overall Results")
                    
                    col1, col2, col3, col4 = st.columns(4, gap="small")
                    
                    with col1:
                        st.metric("Total Games", ml_total)
                    
                    with col2:
                        st.metric("Correct", ml_correct)
                    
                    with col3:
                        st.metric("Wrong", ml_total - ml_correct)
                    
                    with col4:
                        st.metric("Accuracy", f"{ml_accuracy:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # MODEL COMPARISON PAGE
    elif page == "Model Comparison":
        if ml_predictions is None or len(ml_predictions) == 0:
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.info("⚠️ ML Model predictions not available. Train the model first to enable comparison.")
        else:
            st.subheader("🔄 Excel vs ML Model Comparison")
            st.markdown("<br>", unsafe_allow_html=True)
            
            ml_completed = ml_predictions[ml_predictions['ml_correct'].notna()].copy()
            
            if len(ml_completed) == 0:
                col_info, _, _ = st.columns([2, 1, 2])
                with col_info:
                    st.info("No completed games to compare yet")
            else:
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 'YES').sum()
                excel_correct = (ml_completed['excel_correct'] == 'YES').sum()
                
                ml_accuracy = (ml_correct / ml_total * 100)
                excel_accuracy = (excel_correct / ml_total * 100)
                
                col_comparison, _, _ = st.columns([1, 0.3, 1])
                with col_comparison:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Head-to-Head Comparison")
                    st.caption(f"Based on {ml_total} completed games")
                    
                    col1, col2, col3 = st.columns(3, gap="medium")
                    
                    with col1:
                        st.metric("Excel Model Accuracy", f"{excel_accuracy:.1f}%")
                        st.caption(f"{excel_correct}/{ml_total} correct")
                    
                    with col2:
                        st.metric("ML Model Accuracy", f"{ml_accuracy:.1f}%")
                        st.caption(f"{ml_correct}/{ml_total} correct")
                    
                    with col3:
                        diff = ml_accuracy - excel_accuracy
                        st.metric("Difference", f"{diff:+.1f}%")
                        if diff > 0:
                            st.caption("🤖 ML Model leads")
                        elif diff < 0:
                            st.caption("📊 Excel Model leads")
                        else:
                            st.caption("🤝 Tied")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recent games breakdown
                st.markdown("<br>")
                st.markdown("### 📋 Recent Games Breakdown")
                st.markdown("<br>")
                
                recent_games = ml_completed.tail(10)
                
                display_data = []
                for idx, game in recent_games.iterrows():
                    excel_result = "✅" if game['excel_correct'] == 'YES' else "❌"
                    ml_result = "✅" if game['ml_correct'] == 'YES' else "❌"
                    
                    # Get week display - USE week_display_format function
                    week_str = week_display_format(game['week'])
                    
                    display_data.append({
                        "Week": week_str,
                        "Matchup": f"{game['away_team']} @ {game['home_team']}",
                        "Winner": game['actual_winner'],
                        "Excel": f"{excel_result} {game['excel_winner']}",
                        "ML": f"{ml_result} {game['ml_winner']}"
                    })
                
                col_table, _, _ = st.columns([1, 0.3, 1])
                with col_table:
                    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **NFL Prediction Model 2025-26**
    
    📊 **Excel Model**
    Uses HomeEdge Differential and team statistics to predict game outcomes.
    
    🤖 **ML Model** (Optional)
    Machine learning model trained on historical data for enhanced predictions.
    
    **Supports:** Regular Season (Weeks 1-18) and Playoffs (Wild Card, Divisional, Conference, Super Bowl)
    """)

if __name__ == "__main__":
    main()
