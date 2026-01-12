#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Prediction Model - PROPERLY FIXED VERSION
- Reads from "NFL HomeField Model" sheet
- Loads ML predictions from "ML Prediction Model" sheet (not CSV)
- Uses "Locked Correct" for performance calculations
- Fixed date handling for today's games
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
    'wildcard': 19,
    'divisional': 20,
    'conference': 21,
    'superbowl': 22
}

def week_display_format(week_value):
    """Convert week number to display string"""
    if pd.isna(week_value):
        return "Unknown"
    if isinstance(week_value, str):
        week_lower = week_value.lower().strip()
        if week_lower in ROUND_TO_WEEK:
            return week_value.replace('SuperBowl', 'Super Bowl')
        if 'wild' in week_lower:
            return "Wild Card"
        if 'division' in week_lower:
            return "Divisional"
        if 'conference' in week_lower:
            return "Conference"
        if 'super' in week_lower or 'bowl' in week_lower:
            return "Super Bowl"
        return week_value
    try:
        week_int = int(week_value)
        if week_int <= 18:
            return f"Week {week_int}"
        return PLAYOFF_ROUNDS.get(week_int, f"Week {week_int}").replace('SuperBowl', 'Super Bowl')
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
    """Safely convert to int"""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except:
        return default

def safe_float(value, default=0.0):
    """Safely convert to float"""
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

# Custom CSS
st.markdown("""
    <style>
    .main {background: #f5f5f5;padding: 2rem 1rem;}
    .stApp {background: #f5f5f5;}
    h1, h2, h3 {color: #000000 !important;text-align: center;}
    .centered-container {max-width: 1400px;margin: 0 auto;padding: 0 2rem;}
    .game-card {background: #ffffff;padding: 2rem;border-radius: 12px;border: 1px solid #e0e0e0;margin: 1.5rem 0;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);transition: transform 0.2s ease;}
    .game-card:hover {transform: translateY(-2px);box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);}
    .winner, .ml-winner {color: #2e7d32;font-size: 1.5rem;font-weight: 700;text-align: center;padding: 0.75rem;background: #e8f5e9;border-radius: 8px;margin: 0.5rem 0;border: 2px solid #4caf50;}
    .metric-card {background: #ffffff;padding: 1.5rem;border-radius: 12px;border: 1px solid #e0e0e0;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);margin: 1rem 0;}
    .score-display {font-size: 2rem;font-weight: 700;text-align: center;padding: 1rem;background: #f0f0f0;border-radius: 10px;margin: 1rem 0;}
    .prob-badge {display: inline-block;padding: 0.4rem 0.8rem;border-radius: 6px;font-weight: 600;font-size: 0.9rem;}
    </style>
""", unsafe_allow_html=True)

EXCEL_FILE = 'Aidan Conte NFL 2025-26 Prediction Model.xlsx'

def get_probability_color(probability):
    """Convert probability to color gradient"""
    try:
        prob = float(probability)
    except:
        prob = 0.5
    normalized = max(0, min(1, (prob - 0.5) / 0.5))
    if normalized < 0.5:
        t = normalized * 2
        r, g, b = int(211+(251-211)*t), int(47+(192-47)*t), int(47+(45-47)*t)
    else:
        t = (normalized - 0.5) * 2
        r, g, b = int(251+(56-251)*t), int(192+(142-192)*t), int(45+(60-45)*t)
    return f"#{r:02x}{g:02x}{b:02x}", f"rgba({r},{g},{b},0.15)"

@st.cache_data
def load_data():
    """Load data from Excel file"""
    try:
        # Load HomeField Model predictions
        df_raw = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=None)
        home_edge = safe_float(df_raw.iloc[0, 1], 2.140576)
        
        # Load with header
        predictions = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=0)
        
        # Convert date
        predictions['Date'] = pd.to_datetime(predictions['Date'], errors='coerce')
        
        # Normalize week values
        predictions['Week_Num'] = predictions['Week'].apply(normalize_week)
        predictions['Week_Display'] = predictions['Week_Num'].apply(week_display_format)
        
        # Remove rows with no week number
        predictions = predictions[predictions['Week_Num'].notna()].copy()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        standings = standings.fillna({'W': 0, 'L': 0, 'HomeWin%': 0.5, 'AwayWin%': 0.5, 'PTS': 0})
        
        # Load ML predictions from Excel sheet (not CSV)
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            return predictions, standings, home_edge, ml_predictions
        except Exception as e:
            st.warning(f"ML Prediction Model sheet not found: {e}")
            return predictions, standings, home_edge, None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

def display_game(game, standings, home_edge, ml_pred=None):
    """Display a single game"""
    home_team = game['Home Team']
    away_team = game['Away Team']
    
    try:
        home_stats = standings[standings['Team'] == home_team].iloc[0]
        away_stats = standings[standings['Team'] == away_team].iloc[0]
    except:
        st.warning(f"Stats missing for {home_team} or {away_team}")
        return
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**{game['Week_Display']}**")
        with col2:
            try:
                st.markdown(f"**{game['Date'].strftime('%a, %b %d')}**")
            except:
                st.markdown("**TBD**")
        
        st.markdown("---")
        
        # Teams
        col_away, col_vs, col_home = st.columns([2, 1, 2])
        with col_away:
            st.markdown(f"### 🏈 {away_team}")
            st.caption(f"Record: {safe_int(away_stats.get('W',0))}-{safe_int(away_stats.get('L',0))}")
            st.caption(f"Away: {safe_float(away_stats.get('AwayWin%',0.5)):.1%}")
        with col_vs:
            st.markdown("<br><h4 style='text-align:center;'>@</h4>", unsafe_allow_html=True)
        with col_home:
            st.markdown(f"### 🏈 {home_team}")
            st.caption(f"Record: {safe_int(home_stats.get('W',0))}-{safe_int(home_stats.get('L',0))}")
            st.caption(f"Home: {safe_float(home_stats.get('HomeWin%',0.5)):.1%}")
        
        st.markdown("---")
        
        # Excel Prediction
        st.markdown("### 📊 Excel Model")
        excel_winner = game.get('Locked Prediction', game.get('Predictied Winner', 'TBD'))
        excel_prob = safe_float(game.get('Locked Strength of Win', game.get('Strength of Win', 0.5)))
        homefield = safe_float(game.get('Locked HomeField Differential', game.get('HomeEdge Differential', 0)))
        
        prob_color, prob_bg = get_probability_color(excel_prob)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f'<div class="winner">{excel_winner}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="prob-badge" style="background:{prob_bg};color:{prob_color};">{excel_prob:.1%}</span>', unsafe_allow_html=True)
            st.caption(f"HomeField: {homefield:+.2f}")
        
        # ML Prediction
        if ml_pred is not None:
            st.markdown("<br>### 🤖 ML Model", unsafe_allow_html=True)
            ml_winner = ml_pred.get('ml_winner', 'TBD')
            ml_home = safe_int(ml_pred.get('ml_home_score', 0))
            ml_away = safe_int(ml_pred.get('ml_away_score', 0))
            
            try:
                ml_conf_str = str(ml_pred.get('ml_confidence', '0.5'))
                ml_conf = float(ml_conf_str)
                if ml_conf > 1:  # If it's a percentage like 50.0 instead of 0.5
                    ml_conf = ml_conf / 100
            except:
                ml_conf = 0.5
            
            ml_color, ml_bg = get_probability_color(ml_conf)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f'<div class="ml-winner">{ml_winner}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-display">{away_team}: {ml_away} | {home_team}: {ml_home}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span class="prob-badge" style="background:{ml_bg};color:{ml_color};">{ml_conf:.1%}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, home_edge, ml_predictions = load_data()
    
    if predictions is None:
        st.error("Failed to load data. Check Excel file.")
        return
    
    st.sidebar.title("🏈 NFL Predictions")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Today's Games", "All Games", "Performance", "Model Comparison"])
    
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.title("🏈 NFL Prediction Model 2025-26")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # TODAY'S GAMES
    if page == "Today's Games":
        st.subheader("🎯 Today's Games")
        st.markdown("<br>", unsafe_allow_html=True)
        
        today = pd.Timestamp.now().normalize()
        predictions['Date_Only'] = predictions['Date'].dt.normalize()
        todays_games = predictions[predictions['Date_Only'] == today]
        
        if len(todays_games) == 0:
            st.info(f"No games scheduled for {today.strftime('%A, %B %d, %Y')}")
        else:
            st.success(f"📅 {len(todays_games)} game(s) today")
            
            for idx, game in todays_games.iterrows():
                ml_pred = None
                if ml_predictions is not None:
                    try:
                        # Match by week and teams
                        week_num = normalize_week(game['Week'])
                        ml_match = ml_predictions[
                            (ml_predictions['week'] == week_num) &
                            (ml_predictions['home_team'] == game['Home Team']) &
                            (ml_predictions['away_team'] == game['Away Team'])
                        ]
                        if len(ml_match) > 0:
                            ml_pred = ml_match.iloc[0]
                    except Exception as e:
                        pass
                
                display_game(game, standings, home_edge, ml_pred)
    
    # ALL GAMES
    elif page == "All Games":
        st.subheader("📋 All Games")
        st.markdown("<br>", unsafe_allow_html=True)
        
        unique_displays = sorted(predictions['Week_Display'].unique(), 
                                 key=lambda x: predictions[predictions['Week_Display']==x]['Week_Num'].iloc[0])
        selected_week = st.sidebar.selectbox("Filter by Week", ["All Weeks"] + list(unique_displays))
        
        filtered_games = predictions if selected_week == "All Weeks" else predictions[predictions['Week_Display'] == selected_week]
        st.info(f"Showing {len(filtered_games)} games")
        
        for idx, game in filtered_games.iterrows():
            ml_pred = None
            if ml_predictions is not None:
                try:
                    week_num = normalize_week(game['Week'])
                    ml_match = ml_predictions[
                        (ml_predictions['week'] == week_num) &
                        (ml_predictions['home_team'] == game['Home Team']) &
                        (ml_predictions['away_team'] == game['Away Team'])
                    ]
                    if len(ml_match) > 0:
                        ml_pred = ml_match.iloc[0]
                except:
                    pass
            
            display_game(game, standings, home_edge, ml_pred)
    
    # PERFORMANCE
    elif page == "Performance":
        st.subheader("📊 Model Performance")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Excel Model - using LOCKED CORRECT
        correct_col = 'Locked Correct'
        completed = predictions[predictions[correct_col].isin(['YES', 'NO'])].copy()
        
        if len(completed) == 0:
            st.info("No completed games yet")
        else:
            total = len(completed)
            correct = (completed[correct_col] == 'YES').sum()
            accuracy = (correct / total * 100)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Excel Model - Overall Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Games", total)
            col2.metric("Correct", correct)
            col3.metric("Wrong", total - correct)
            col4.metric("Accuracy", f"{accuracy:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Week-by-week
            st.markdown("<br>")
            week_stats = completed.groupby('Week_Display').agg({
                correct_col: [('correct', lambda x: (x == 'YES').sum()), ('total', 'count')]
            }).reset_index()
            week_stats.columns = ['Week', 'Correct', 'Total']
            week_stats['Accuracy'] = (week_stats['Correct'] / week_stats['Total'] * 100).round(1)
            week_stats = week_stats.merge(
                completed[['Week_Display', 'Week_Num']].drop_duplicates(),
                left_on='Week', right_on='Week_Display'
            ).sort_values('Week_Num').drop(columns=['Week_Display', 'Week_Num'])
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Week-by-Week Performance")
            st.dataframe(week_stats, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ML Model Performance
        if ml_predictions is not None:
            st.markdown("---### 🤖 ML Model Performance")
            ml_completed = ml_predictions[ml_predictions['ml_correct'].isin(['YES', 'NO'])].copy()
            
            if len(ml_completed) > 0:
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 'YES').sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 Overall Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", ml_total)
                col2.metric("Correct", ml_correct)
                col3.metric("Wrong", ml_total - ml_correct)
                col4.metric("Accuracy", f"{ml_accuracy:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # MODEL COMPARISON
    elif page == "Model Comparison":
        if ml_predictions is None or len(ml_predictions) == 0:
            st.info("ML predictions not available")
        else:
            st.subheader("🔄 Excel vs ML Comparison")
            st.markdown("<br>", unsafe_allow_html=True)
            
            ml_completed = ml_predictions[
                (ml_predictions['ml_correct'].isin(['YES', 'NO'])) &
                (ml_predictions['excel_correct'].isin(['YES', 'NO']))
            ].copy()
            
            if len(ml_completed) == 0:
                st.info("No completed games to compare")
            else:
                ml_total = len(ml_completed)
                ml_correct = (ml_completed['ml_correct'] == 'YES').sum()
                excel_correct = (ml_completed['excel_correct'] == 'YES').sum()
                ml_accuracy = (ml_correct / ml_total * 100)
                excel_accuracy = (excel_correct / ml_total * 100)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Head-to-Head Comparison")
                st.caption(f"Based on {ml_total} completed games")
                col1, col2, col3 = st.columns(3)
                col1.metric("Excel Accuracy", f"{excel_accuracy:.1f}%")
                col1.caption(f"{excel_correct}/{ml_total} correct")
                col2.metric("ML Accuracy", f"{ml_accuracy:.1f}%")
                col2.caption(f"{ml_correct}/{ml_total} correct")
                diff = ml_accuracy - excel_accuracy
                col3.metric("Difference", f"{diff:+.1f}%")
                col3.caption("🤖 ML leads" if diff > 0 else "📊 Excel leads" if diff < 0 else "🤝 Tied")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recent games
                st.markdown("### 📋 Recent Games Breakdown")
                recent = ml_completed.tail(10)
                display_data = []
                for _, game in recent.iterrows():
                    excel_result = "✅" if game['excel_correct'] == 'YES' else "❌"
                    ml_result = "✅" if game['ml_correct'] == 'YES' else "❌"
                    week_str = week_display_format(game['week'])
                    display_data.append({
                        "Week": week_str,
                        "Matchup": f"{game['away_team']} @ {game['home_team']}",
                        "Winner": game['actual_winner'],
                        "Excel": f"{excel_result} {game['excel_winner']}",
                        "ML": f"{ml_result} {game['ml_winner']}"
                    })
                st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---### About")
    st.sidebar.info("**NFL Prediction Model 2025-26**\n\n📊 Excel & 🤖 ML Models\n\nRegular Season + Playoffs")

if __name__ == "__main__":
    main()
