#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:29:19 2025

@author: aidanconte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Prediction Model - Web App Version with ML Integration
Streamlit web interface for NFL game predictions
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

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
    normalized = (probability - 0.5) / 0.5
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
    """Load data from Excel file"""
    try:
        # Load predictions
        df_raw = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=None)
        home_edge = df_raw.iloc[0, 1]  # HomeEdge from B1
        
        df_raw_with_header = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=0)
        predictions = df_raw_with_header.iloc[:, 3:].reset_index(drop=True)
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Try to load ML predictions if available
        try:
            ml_predictions = pd.read_csv('nfl_ml_tracker_with_scores.csv')
            ml_predictions['date'] = pd.to_datetime(ml_predictions['date'])
        except:
            ml_predictions = None
        
        return predictions, standings, home_edge, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def display_game(game, standings, home_edge, ml_pred=None):
    """Display a single game card"""
    home_team = game['Home Team']
    away_team = game['Away Team']
    game_date = game['Date']
    week = game['Week']
    
    predicted_winner = game['Predictied Winner']  # Note: typo in Excel
    strength = game['Strength of Win']
    homeice_diff = game['HomeEdge Differential']
    
    # Get team records
    home_row = standings[standings['Team'] == home_team]
    away_row = standings[standings['Team'] == away_team]
    
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    
    # Game header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🏈 Week {week}")
        st.caption(game_date.strftime('%A, %B %d, %Y'))
    
    # Teams
    col_away, col_vs, col_home = st.columns([2, 1, 2])
    
    with col_away:
        st.markdown('<div class="team-display">', unsafe_allow_html=True)
        st.markdown(f"**{away_team}**")
        if len(away_row) > 0:
            away_w = int(away_row['W'].values[0])
            away_l = int(away_row['L'].values[0])
            away_ties = int(away_row['Ties'].values[0])
            st.caption(f"{away_w}-{away_l}-{away_ties}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_vs:
        st.markdown('<div style="text-align: center; padding-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("**@**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_home:
        st.markdown('<div class="team-display">', unsafe_allow_html=True)
        st.markdown(f"**{home_team}**")
        if len(home_row) > 0:
            home_w = int(home_row['W'].values[0])
            home_l = int(home_row['L'].values[0])
            home_ties = int(home_row['Ties'].values[0])
            st.caption(f"{home_w}-{home_l}-{home_ties}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Excel Model Prediction
    st.markdown("---")
    st.markdown("#### 📊 Excel Model Prediction")
    
    # Get probability color
    prob_color, prob_bg = get_probability_color(strength if pd.notna(strength) else 0.5)
    
    st.markdown(
        f'<div class="winner">🏆 Predicted Winner: {predicted_winner}</div>',
        unsafe_allow_html=True
    )
    
    if pd.notna(strength) and strength > 0:
        st.markdown(
            f'<div style="text-align: center; margin: 0.5rem 0;">'
            f'<span class="prob-badge" style="background: {prob_bg}; color: {prob_color};">'
            f'Win Probability: {strength:.1%}</span></div>',
            unsafe_allow_html=True
        )
    
    # Score prediction
    if len(home_row) > 0 and len(away_row) > 0:
        home_offense = home_row['HomePts per Game'].values[0]
        home_defense = home_row['HomePts Against'].values[0]
        away_offense = away_row['AwayPts per Game'].values[0]
        away_defense = away_row['AwayPts Against'].values[0]
        
        predicted_home_raw = (home_offense + away_defense) / 2 + (home_edge / 2)
        predicted_away_raw = (away_offense + home_defense) / 2 - (home_edge / 2)
        
        predicted_home = round(predicted_home_raw)
        predicted_away = round(predicted_away_raw)
        
        # Ensure winner has higher score
        if predicted_winner == home_team and predicted_home <= predicted_away:
            predicted_home = predicted_away + 1
        elif predicted_winner == away_team and predicted_away <= predicted_home:
            predicted_away = predicted_home + 1
        
        st.markdown(
            f'<div class="score-display">{away_team} {predicted_away} - {predicted_home} {home_team}</div>',
            unsafe_allow_html=True
        )
    
    # HomeEdge info
    st.markdown(
        f'<div style="text-align: center; margin-top: 0.5rem;">'
        f'<span class="diff-badge" style="background: #f5f5f5; color: #666;">HomeEdge Diff: {homeice_diff:+.2f}</span>'
        f'<span class="diff-badge" style="background: #f5f5f5; color: #666; margin-left: 0.5rem;">HomeEdge: {home_edge:+.2f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # ML Model Prediction if available
    if ml_pred is not None:
        st.markdown("---")
        st.markdown("#### 🤖 ML Model Prediction")
        
        ml_prob_color, ml_prob_bg = get_probability_color(ml_pred['ml_confidence'])
        
        st.markdown(
            f'<div class="ml-winner">🏆 Predicted Winner: {ml_pred["ml_predicted_winner"]}</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            f'<div style="text-align: center; margin: 0.5rem 0;">'
            f'<span class="prob-badge" style="background: {ml_prob_bg}; color: {ml_prob_color};">'
            f'Win Probability: {ml_pred["ml_confidence"]:.1%}</span></div>',
            unsafe_allow_html=True
        )
        
        if 'ml_home_score' in ml_pred and 'ml_away_score' in ml_pred:
            st.markdown(
                f'<div class="score-display">{away_team} {ml_pred["ml_away_score"]} - {ml_pred["ml_home_score"]} {home_team}</div>',
                unsafe_allow_html=True
            )
        
        # Agreement indicator
        excel_winner = predicted_winner
        ml_winner = ml_pred['ml_predicted_winner']
        
        if excel_winner == ml_winner:
            st.success("✅ Both models agree on winner")
        else:
            st.warning(f"⚠️ Models disagree - Excel: {excel_winner}, ML: {ml_winner}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Load data
    predictions, standings, home_edge, ml_predictions = load_data()
    
    if predictions is None:
        st.error("Failed to load data. Please check that the Excel file exists.")
        return
    
    # Title
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.title("🏈 NFL Prediction Model 2025-26")
    st.markdown(f"<p style='text-align: center; color: #666;'>HomeEdge: {home_edge:+.2f} points</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Today's Games", "All Games", "Performance", "Model Comparison"])
    
    # TODAY'S GAMES PAGE
    if page == "Today's Games":
        st.subheader("📅 Today's Predictions")
        st.markdown("<br>", unsafe_allow_html=True)
        
        today = datetime.now().date()
        todays_games = predictions[predictions['Date'].dt.date == today]
        
        if len(todays_games) == 0:
            # Show next 7 days
            week_end = today + pd.Timedelta(days=7)
            upcoming_games = predictions[
                (predictions['Date'].dt.date >= today) & 
                (predictions['Date'].dt.date <= week_end)
            ].sort_values('Date')
            
            if len(upcoming_games) > 0:
                col_info, _, _ = st.columns([2, 1, 2])
                with col_info:
                    st.info(f"No games today. Showing {len(upcoming_games)} upcoming games in next 7 days")
                
                for idx, game in upcoming_games.iterrows():
                    # Find ML prediction if exists
                    ml_pred = None
                    if ml_predictions is not None:
                        ml_match = ml_predictions[
                            (ml_predictions['home_team'] == game['Home Team']) &
                            (ml_predictions['away_team'] == game['Away Team']) &
                            (ml_predictions['date'].dt.date == game['Date'].date())
                        ]
                        if len(ml_match) > 0:
                            ml_pred = ml_match.iloc[0]
                    
                    display_game(game, standings, home_edge, ml_pred)
            else:
                col_info, _, _ = st.columns([2, 1, 2])
                with col_info:
                    st.warning("No games scheduled in the next 7 days")
        else:
            col_info, _, _ = st.columns([2, 1, 2])
            with col_info:
                st.success(f"🏈 Found {len(todays_games)} game(s) today!")
            
            for idx, game in todays_games.iterrows():
                # Find ML prediction if exists
                ml_pred = None
                if ml_predictions is not None:
                    ml_match = ml_predictions[
                        (ml_predictions['home_team'] == game['Home Team']) &
                        (ml_predictions['away_team'] == game['Away Team']) &
                        (ml_predictions['date'].dt.date == game['Date'].date())
                    ]
                    if len(ml_match) > 0:
                        ml_pred = ml_match.iloc[0]
                
                display_game(game, standings, home_edge, ml_pred)
    
    # ALL GAMES PAGE
    elif page == "All Games":
        st.subheader("📋 All Games")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Week filter
        weeks = sorted(predictions['Week'].unique())
        selected_week = st.sidebar.selectbox("Filter by Week", ["All Weeks"] + [f"Week {w}" for w in weeks])
        
        if selected_week == "All Weeks":
            filtered_games = predictions
        else:
            week_num = int(selected_week.split()[1])
            filtered_games = predictions[predictions['Week'] == week_num]
        
        col_info, _, _ = st.columns([2, 1, 2])
        with col_info:
            st.info(f"Showing {len(filtered_games)} games")
        
        for idx, game in filtered_games.iterrows():
            # Find ML prediction if exists
            ml_pred = None
            if ml_predictions is not None:
                ml_match = ml_predictions[
                    (ml_predictions['home_team'] == game['Home Team']) &
                    (ml_predictions['away_team'] == game['Away Team']) &
                    (ml_predictions['date'].dt.date == game['Date'].date())
                ]
                if len(ml_match) > 0:
                    ml_pred = ml_match.iloc[0]
            
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
            
            # Week-by-week
            st.markdown("<br>", unsafe_allow_html=True)
            week_stats = completed.groupby('Week').agg({
                correct_col: [
                    ('correct', lambda x: (x == 'YES').sum()),
                    ('total', 'count')
                ]
            }).reset_index()
            week_stats.columns = ['Week', 'Correct', 'Total']
            week_stats['Accuracy'] = (week_stats['Correct'] / week_stats['Total'] * 100).round(1)
            
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
                ml_correct = (ml_completed['ml_correct'] == 1).sum()
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
                ml_correct = (ml_completed['ml_correct'] == 1).sum()
                excel_correct = (ml_completed['excel_correct'] == 1).sum()
                
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
                    excel_result = "✅" if game['excel_correct'] == 1 else "❌"
                    ml_result = "✅" if game['ml_correct'] == 1 else "❌"
                    
                    display_data.append({
                        "Date": game['date'].strftime('%Y-%m-%d'),
                        "Matchup": f"{game['away_team']} @ {game['home_team']}",
                        "Winner": game['actual_winner'],
                        "Excel": f"{excel_result} {game['excel_predicted_winner']}",
                        "ML": f"{ml_result} {game['ml_predicted_winner']}"
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
    
    Model incorporates:
    - Home field advantage
    - Team offensive/defensive stats
    - Win/loss records
    - Historical performance
    """)

if __name__ == "__main__":
    main()
