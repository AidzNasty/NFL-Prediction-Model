#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Prediction Model - Web App
Displays Excel and ML predictions side by side in DraftKings/FanDuel style
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NFL Prediction Model 2025-26",
    page_icon="🏈",
    layout="wide"
)

# DraftKings/FanDuel Style CSS
st.markdown("""
    <style>
    /* Main styling - Dark theme like DraftKings/FanDuel */
    .main {
        background: #0d1117;
        padding: 1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark background for entire app */
    .stApp {
        background: #0d1117;
    }
    
    /* Typography - White text on dark */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Radio buttons styled */
    .stRadio > div {
        background: #161b22;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stRadio label {
        color: #c9d1d9;
        font-weight: 500;
    }
    
    /* Buttons - Green accent like DraftKings */
    .stButton > button {
        background: #00d4aa;
        color: #0d1117;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #00b894;
        transform: translateY(-1px);
    }
    
    /* Selectbox dark theme */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stSelectbox label {
        color: #c9d1d9 !important;
        font-weight: 500;
    }
    
    .stSelectbox > div > div {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    /* Dataframe dark theme */
    .dataframe {
        background: #161b22;
        color: #c9d1d9;
    }
    
    /* Caption text */
    .stCaption {
        color: #8b949e;
    }
    
    /* Info boxes */
    .stInfo {
        background: #161b22;
        border-left: 4px solid #00d4aa;
        color: #c9d1d9;
    }
    
    /* Betting Card Style */
    .betting-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .betting-card:hover {
        border-color: #00d4aa;
        box-shadow: 0 0 0 1px #00d4aa;
    }
    
    /* Playoff Badge */
    .playoff-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    /* Model Badge */
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-excel {
        background: #1f6feb;
        color: #ffffff;
    }
    
    .badge-ml {
        background: #a371f7;
        color: #ffffff;
    }
    
    /* Winner Highlight */
    .winner-highlight {
        color: #00d4aa;
        font-weight: 700;
    }
    
    /* Stats Grid */
    .stat-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4aa;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.5rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NFL 2025-26 Prediction Model.xlsx'

PLAYOFF_ROUNDS = {
    19: 'Wild Card',
    20: 'Divisional',
    21: 'Conference Championship',
    22: 'Super Bowl'
}

def normalize_week(week_value):
    """Convert week to standard format"""
    if pd.isna(week_value):
        return None
    if isinstance(week_value, (int, float)):
        return int(week_value)
    week_str = str(week_value).strip().lower()
    playoff_map = {
        'wildcard': 19, 'wild card': 19,
        'division': 20, 'divisional': 20,
        'confchamp': 21, 'conference': 21,
        'superbowl': 22, 'super bowl': 22
    }
    for key, value in playoff_map.items():
        if key in week_str:
            return value
    try:
        return int(float(week_str))
    except:
        return None

def week_display_format(week_num):
    """Display week nicely"""
    if pd.isna(week_num):
        return "Unknown"
    week_num = int(week_num)
    if week_num <= 18:
        return f"Week {week_num}"
    return PLAYOFF_ROUNDS.get(week_num, f"Week {week_num}")

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel data"""
    try:
        # Load Excel predictions
        excel_raw = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=None)
        excel = excel_raw.iloc[:, 3:].reset_index(drop=True)
        excel.columns = excel.iloc[0]
        excel = excel[1:].reset_index(drop=True)
        
        # Filter bad rows
        excel = excel[
            pd.notna(excel['Week']) & 
            pd.notna(excel['Home Team']) & 
            pd.notna(excel['Away Team'])
        ].copy()
        
        # Normalize week
        excel['Week_Num'] = excel['Week'].apply(normalize_week)
        excel = excel[excel['Week_Num'].notna()].copy()
        excel['Week_Display'] = excel['Week_Num'].apply(week_display_format)
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        standings = standings.fillna({'W': 0, 'L': 0, 'HomeWin%': 0.5, 'AwayWin%': 0.5})
        
        # Load ML predictions from Excel sheet "ML Prediction Model" (columns B-G)
        ml_predictions = None
        try:
            ml_raw = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=None)
            
            # Extract columns B through G (indices 1-6)
            ml_data = ml_raw.iloc[:, 1:7].copy()
            
            # Row 0 has headers
            ml_data.columns = ml_data.iloc[0]
            ml_data = ml_data[1:].reset_index(drop=True)
            
            ml_predictions = ml_data
            
            st.sidebar.success(f"✅ Loaded {len(ml_predictions)} ML predictions from Excel")
            if len(ml_predictions) > 0:
                st.sidebar.caption(f"Columns: {', '.join(ml_predictions.columns.tolist())}")
                # Show first few games
                st.sidebar.caption(f"Sample: {ml_predictions.iloc[0]['away_team']} @ {ml_predictions.iloc[0]['home_team']}")
        except Exception as e:
            st.sidebar.error(f"ML sheet error: {e}")
            import traceback
            st.sidebar.caption(traceback.format_exc())
            ml_predictions = pd.DataFrame()
        
        return excel, standings, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def get_ml_prediction(home_team, away_team, week_num, ml_predictions, debug=False):
    """Get ML model prediction - match by teams only"""
    if ml_predictions is None or len(ml_predictions) == 0:
        return None
    
    try:
        # Match by home and away team only (no week in ML sheet)
        ml_game = ml_predictions[
            (ml_predictions['home_team'] == home_team) &
            (ml_predictions['away_team'] == away_team)
        ]
        
        if debug:
            if len(ml_game) > 0:
                st.sidebar.success(f"✓ Found: {away_team} @ {home_team}")
            else:
                st.sidebar.warning(f"✗ Not found: {away_team} @ {home_team}")
        
    except Exception as e:
        if debug:
            st.sidebar.error(f"ML match error: {e}")
        return None
    
    if len(ml_game) == 0:
        return None
    
    ml_game = ml_game.iloc[0]
    
    # Get confidence
    ml_confidence = ml_game.get('ml_confidence', 0.5)
    if isinstance(ml_confidence, str) and '%' in str(ml_confidence):
        ml_confidence = float(str(ml_confidence).strip('%')) / 100.0
    else:
        try:
            ml_confidence = float(ml_confidence) if pd.notna(ml_confidence) else 0.5
        except:
            ml_confidence = 0.5
    
    # Get scores
    try:
        home_score = int(float(ml_game.get('ml_home_score', 0)))
        away_score = int(float(ml_game.get('ml_away_score', 0)))
    except:
        home_score = 0
        away_score = 0
    
    return {
        'winner': str(ml_game.get('ml_winner', 'Unknown')),
        'home_score': home_score,
        'away_score': away_score,
        'confidence': ml_confidence
    }

def display_game_pick(game_row, standings, ml_predictions):
    """Display a single game's pick in betting card style"""
    home_team = game_row['Home Team']
    away_team = game_row['Away Team']
    week_display = game_row['Week_Display']
    week_num = game_row['Week_Num']
    is_playoff = week_num > 18
    
    # Get Excel prediction
    excel_winner = game_row.get('Locked Prediction', game_row.get('Predicted Winner', 'TBD'))
    excel_confidence = float(game_row.get('Locked Strength of Win', game_row.get('Strength of Win', 0.5)))
    excel_homefield = float(game_row.get('Locked HomeField Differential', game_row.get('HomeEdge Differential', 0)))
    
    # Get ML prediction
    ml_pred = get_ml_prediction(home_team, away_team, week_num, ml_predictions)
    
    # Get team records
    try:
        home_row = standings[standings['Team'] == home_team].iloc[0]
        away_row = standings[standings['Team'] == away_team].iloc[0]
    except:
        return
    
    st.markdown('<div class="betting-card">', unsafe_allow_html=True)
    
    # Playoff badge
    if is_playoff:
        st.markdown(f'<div class="playoff-badge">🏆 {week_display}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color: #8b949e; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem; text-transform: uppercase;">{week_display}</div>', unsafe_allow_html=True)
    
    # Teams
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff;">{away_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem;">{int(away_row["W"])}-{int(away_row["L"])} | Away: {away_row["AwayWin%"]:.1%}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; text-align: center; padding: 0.5rem 0;">@</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #ffffff; text-align: right;">{home_team}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem; text-align: right;">{int(home_row["W"])}-{int(home_row["L"])} | Home: {home_row["HomeWin%"]:.1%}</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
    
    # Excel Prediction
    st.markdown('<span class="model-badge badge-excel">EXCEL MODEL</span>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{excel_winner}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>HomeField Edge</div>
                <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_homefield:+.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem;'>
                <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                <div style='color: #00d4aa; font-size: 1.1rem; margin-top: 0.25rem;'>{excel_confidence:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # ML Prediction
    if ml_pred:
        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        st.markdown('<span class="model-badge badge-ml">ML MODEL</span>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Pick</div>
                    <div class='winner-highlight' style='font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['winner']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Predicted Score</div>
                    <div style='color: #ffffff; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['away_score']}-{ml_pred['home_score']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'>
                    <div style='color: #8b949e; font-size: 0.75rem; text-transform: uppercase;'>Confidence</div>
                    <div style='color: #a371f7; font-size: 1.1rem; margin-top: 0.25rem;'>{ml_pred['confidence']:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Agreement indicator
        if excel_winner == ml_pred['winner']:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #00d4aa; font-weight: 600;">✓ Both Models Agree</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; margin-top: 1rem; padding: 0.5rem; background: #0d1117; border-radius: 4px; color: #ffa500; font-weight: 600;">⚠ Models Disagree</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, ml_predictions = load_data()
    
    if predictions is None or standings is None:
        st.error("❌ Error loading data. Please check your Excel file.")
        return
    
    # Header
    st.markdown("""
        <div style='background: #161b22; border-bottom: 2px solid #00d4aa; padding: 1.5rem 0; margin-bottom: 2rem;'>
            <div style='max-width: 1200px; margin: 0 auto; padding: 0 1rem;'>
                <h1 style='color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700;'>NFL PREDICTIONS</h1>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>2025-26 Season • Advanced Analytics & Machine Learning</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    now = datetime.now()
    st.caption(f"Last updated: {now.strftime('%Y-%m-%d %I:%M %p')} ET")
    
    # Top Dropdown Menu
    st.markdown("""
        <div style='background: #161b22; border: 1px solid #30363d; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;'>
            <p style='color: #c9d1d9; font-weight: 600; margin: 0 0 0.5rem 0; font-size: 0.9rem;'>📊 SELECT PAGE</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        label="Navigate to:",
        options=["🏈 Upcoming Games", "📋 All Games", "📊 Standings", "📈 Model Performance"],
        index=0,
        label_visibility="collapsed",
        key="page_selector"
    )
    
    # Sidebar
    st.sidebar.markdown("""
        <div style='background: #161b22; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #30363d;'>
            <h3 style='color: #ffffff; margin: 0; font-size: 1rem;'>QUICK ACTIONS</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # PAGE CONTENT
    if page == "🏈 Upcoming Games":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>UPCOMING GAMES</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Next games to be played</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get upcoming games (no Locked Correct marked)
        upcoming_games = predictions[
            predictions['Locked Correct'].isna() | 
            (~predictions['Locked Correct'].isin(['YES', 'NO']))
        ].copy()
        
        if len(upcoming_games) == 0:
            st.info("✅ No upcoming games - Season complete!")
        else:
            st.success(f"🏈 {len(upcoming_games)} upcoming games")
            
            for idx, game in upcoming_games.iterrows():
                display_game_pick(game, standings, ml_predictions)
    
    elif page == "📋 All Games":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>ALL GAMES</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Full season schedule and predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        unique_displays = sorted(predictions['Week_Display'].unique(), 
                                 key=lambda x: predictions[predictions['Week_Display']==x]['Week_Num'].iloc[0])
        selected_week = st.selectbox("Filter by Week", ["All Weeks"] + list(unique_displays))
        
        filtered_games = predictions if selected_week == "All Weeks" else predictions[predictions['Week_Display'] == selected_week]
        st.info(f"Showing {len(filtered_games)} games")
        
        for idx, game in filtered_games.iterrows():
            display_game_pick(game, standings, ml_predictions)
    
    elif page == "📊 Standings":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>STANDINGS</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>2025-26 Season Team Rankings</p>
            </div>
        """, unsafe_allow_html=True)
        
        if standings is not None and len(standings) > 0:
            display_cols = ['Team', 'W', 'L', 'HomeWin%', 'AwayWin%', 'HomePts per Game', 'AwayPts per Game']
            available_cols = [col for col in display_cols if col in standings.columns]
            
            if len(available_cols) > 0:
                standings_display = standings[available_cols].copy()
                
                if 'HomeWin%' in standings_display.columns:
                    standings_display['Home Win %'] = standings_display['HomeWin%'].apply(lambda x: f"{x:.1%}")
                    standings_display = standings_display.drop('HomeWin%', axis=1)
                if 'AwayWin%' in standings_display.columns:
                    standings_display['Away Win %'] = standings_display['AwayWin%'].apply(lambda x: f"{x:.1%}")
                    standings_display = standings_display.drop('AwayWin%', axis=1)
                
                st.dataframe(
                    standings_display,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            else:
                st.dataframe(standings, use_container_width=True, hide_index=True, height=600)
    
    elif page == "📈 Model Performance":
        st.markdown("""
            <div style='background: #161b22; border: 1px solid #30363d; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;'>
                <h2 style='color: #ffffff; margin: 0; font-size: 1.5rem;'>MODEL PERFORMANCE</h2>
                <p style='color: #8b949e; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Track accuracy and compare predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Excel Model Performance
        st.markdown("### Excel Model Performance")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            wrong = total - correct
            accuracy = (correct / total * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Games</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{correct}</div>
                        <div class="stat-label">Correct</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="color: #f85149;">{wrong}</div>
                        <div class="stat-label">Incorrect</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{accuracy:.1f}%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No completed games yet")
        
        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
        
        # ML Model Performance
        st.markdown("### ML Model Performance")
        completed = predictions[predictions['ml_correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['ml_correct'] == 'YES').sum()
            wrong = total - correct
            accuracy = (correct / total * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{total}</div>
                        <div class="stat-label">Total Games</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{correct}</div>
                        <div class="stat-label">Correct</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="color: #f85149;">{wrong}</div>
                        <div class="stat-label">Incorrect</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{accuracy:.1f}%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No completed games yet")

if __name__ == "__main__":
    main()
