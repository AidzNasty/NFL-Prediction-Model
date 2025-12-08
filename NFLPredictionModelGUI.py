#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Prediction Model - Simplified Web App
Displays Excel and ML predictions side by side
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Page config
st.set_page_config(
    page_title="NFL Prediction Model 2025-26",
    page_icon="🏈",
    layout="wide"
)

# Simplified CSS
st.markdown("""
    <style>
    .main {
        background: #ffffff;
        padding: 2rem;
    }
    
    h1, h2, h3 {
        color: #1a1a1a;
    }
    
    .game-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .winner-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #28a745;
    }
    
    .score-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #495057;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EXCEL_FILE = 'Aidan Conte NFL 2025-26 Prediction Model.xlsx'
EASTERN = pytz.timezone('America/New_York')

@st.cache_data(ttl=3600)
def load_data():
    """Load Excel and ML data"""
    try:
        # Load Excel predictions
        df_raw = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=None)
        home_edge = df_raw.iloc[0, 1]
        
        df_raw_with_header = pd.read_excel(EXCEL_FILE, sheet_name='NFL HomeField Model', header=0)
        predictions = df_raw_with_header.iloc[:, 3:].reset_index(drop=True)
        predictions['Date'] = pd.to_datetime(predictions['Date']).dt.normalize()
        
        # Load standings
        standings = pd.read_excel(EXCEL_FILE, sheet_name='Standings')
        
        # Load ML predictions
        ml_predictions = None
        try:
            ml_predictions = pd.read_excel(EXCEL_FILE, sheet_name='ML Prediction Model', header=0)
            
            if len(ml_predictions) > 0:
                # Rename columns to match expected format
                rename_map = {
                    'ml_winner': 'ml_predicted_winner',
                    'ml_home_score': 'ml_predicted_home_score',
                    'ml_away_score': 'ml_predicted_away_score',
                    'ml_confidence': 'ml_confidence'
                }
                ml_predictions = ml_predictions.rename(columns={k: v for k, v in rename_map.items() if k in ml_predictions.columns})
                
                # Convert percentage strings
                if 'ml_confidence' in ml_predictions.columns:
                    ml_predictions['ml_confidence'] = ml_predictions['ml_confidence'].apply(
                        lambda x: float(str(x).strip('%')) / 100.0 if isinstance(x, str) and '%' in str(x) else float(x) if pd.notna(x) else 0.5
                    )
                
                # Convert YES/NO to 1/0 for correctness columns
                for col in ['ml_correct', 'excel_correct']:
                    if col in ml_predictions.columns:
                        ml_predictions[col] = ml_predictions[col].apply(
                            lambda x: 1 if str(x).upper() == 'YES' else 0 if str(x).upper() == 'NO' else np.nan
                        )
                
                st.sidebar.success(f"ML Model loaded: {len(ml_predictions)} predictions")
            else:
                st.sidebar.warning("ML Prediction Model sheet is empty")
                ml_predictions = pd.DataFrame()
                
        except Exception as e:
            st.sidebar.error(f"ML Model loading failed: {str(e)}")
            ml_predictions = pd.DataFrame()
        
        return predictions, standings, home_edge, ml_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def calculate_excel_prediction(home_team, away_team, standings, predictions, game_week, home_edge):
    """Calculate Excel model prediction"""
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    # Get prediction from sheet if available
    predicted_winner = None
    strength = None
    homeice_diff = None
    
    if predictions is not None and game_week is not None:
        try:
            game_match = predictions[
                (predictions['Home Team'] == home_team) & 
                (predictions['Away Team'] == away_team) &
                (predictions['Week'] == game_week)
            ]
            
            if len(game_match) > 0:
                game = game_match.iloc[0]
                predicted_winner = game['Predictied Winner']  # Note: typo in Excel
                strength = game['Strength of Win']
                homeice_diff = game['HomeEdge Differential']
        except:
            pass
    
    # Calculate if not found
    if predicted_winner is None:
        home_home_win_pct = home_row['HomeWin%']
        away_away_win_pct = away_row['AwayWin%']
        homeice_diff = (home_home_win_pct - away_away_win_pct) * home_edge
        predicted_winner = home_team if homeice_diff > 0 else away_team
        strength = 0.5 + (abs(homeice_diff) / (2 * home_edge))
        strength = min(0.85, max(0.52, strength))
    
    # Calculate scores
    home_offense = home_row['HomePts per Game']
    home_defense = home_row['HomePts Against']
    away_offense = away_row['AwayPts per Game']
    away_defense = away_row['AwayPts Against']
    
    predicted_home_raw = (home_offense + away_defense) / 2 + (home_edge / 2)
    predicted_away_raw = (away_offense + home_defense) / 2 - (home_edge / 2)
    
    predicted_home = max(7, min(45, round(predicted_home_raw)))
    predicted_away = max(7, min(45, round(predicted_away_raw)))
    
    # Ensure winner has more points
    if predicted_winner == home_team and predicted_home <= predicted_away:
        predicted_home = predicted_away + 3
    elif predicted_winner == away_team and predicted_away <= predicted_home:
        predicted_away = predicted_home + 3
    
    return {
        'winner': predicted_winner,
        'home_score': predicted_home,
        'away_score': predicted_away,
        'confidence': strength if strength else 0.5,
        'homeice_diff': homeice_diff if homeice_diff else 0
    }

def get_ml_prediction(home_team, away_team, game_week, ml_predictions):
    """Get ML model prediction"""
    if ml_predictions is None or len(ml_predictions) == 0:
        return None
    
    try:
        ml_game = ml_predictions[
            (ml_predictions['home_team'] == home_team) & 
            (ml_predictions['away_team'] == away_team) &
            (ml_predictions['week'] == game_week)
        ]
    except:
        return None
    
    if len(ml_game) == 0:
        return None
    
    ml_game = ml_game.iloc[0]
    
    # Get scores
    try:
        ml_home_score = int(float(ml_game.get('ml_predicted_home_score', 21)))
        ml_away_score = int(float(ml_game.get('ml_predicted_away_score', 17)))
    except:
        ml_home_score = 21
        ml_away_score = 17
    
    ml_confidence = ml_game.get('ml_confidence', 0.5)
    
    return {
        'winner': ml_game['ml_predicted_winner'],
        'home_score': ml_home_score,
        'away_score': ml_away_score,
        'confidence': ml_confidence
    }

def display_game(game, standings, predictions, home_edge, ml_predictions):
    """Display a single game prediction"""
    home_team = game['Home Team']
    away_team = game['Away Team']
    game_date = game['Date']
    week = game['Week']
    
    # Get predictions
    excel_pred = calculate_excel_prediction(home_team, away_team, standings, predictions, week, home_edge)
    ml_pred = get_ml_prediction(home_team, away_team, week, ml_predictions)
    
    # Get team records
    home_row = standings[standings['Team'] == home_team].iloc[0]
    away_row = standings[standings['Team'] == away_team].iloc[0]
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        # Header
        st.markdown(f"**Week {int(week)} - {game_date.strftime('%B %d, %Y')}**")
        
        # Teams
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {away_team}")
            st.caption(f"Record: {int(away_row['W'])}-{int(away_row['L'])}-{int(away_row['Ties'])}")
        with col2:
            st.markdown(f"### {home_team}")
            st.caption(f"Record: {int(home_row['W'])}-{int(home_row['L'])}-{int(home_row['Ties'])}")
        
        st.divider()
        
        # Predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("**Excel Model**")
            st.markdown(f'<div class="winner-text">Winner: {excel_pred["winner"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">Score: {excel_pred["away_score"]}-{excel_pred["home_score"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Confidence: {excel_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">HomeEdge Diff: {excel_pred["homeice_diff"]:+.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("**ML Model**")
            if ml_pred:
                st.markdown(f'<div class="winner-text">Winner: {ml_pred["winner"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-text">Score: {ml_pred["away_score"]}-{ml_pred["home_score"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Confidence: {ml_pred["confidence"]:.1%}</div>', unsafe_allow_html=True)
            else:
                st.info("No ML prediction available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agreement
        if ml_pred and excel_pred['winner'] == ml_pred['winner']:
            st.success("Both models agree on winner")
        elif ml_pred:
            st.warning("Models predict different winners")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    predictions, standings, home_edge, ml_predictions = load_data()
    
    if predictions is None or standings is None:
        return
    
    st.title("NFL Prediction Model 2025-26")
    eastern_now = datetime.now(EASTERN)
    st.caption(f"Last updated: {eastern_now.strftime('%Y-%m-%d %I:%M:%S %p')} ET")
    st.caption(f"HomeEdge: {home_edge:+.2f} points")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    page = st.sidebar.radio("Select Page", ["This Week's Games", "Custom Matchup", "Performance"])
    
    if page == "This Week's Games":
        # Get current week from upcoming games
        eastern_now = datetime.now(EASTERN)
        today = pd.Timestamp(eastern_now.date()).normalize()
        
        # Find the current/next week
        upcoming = predictions[predictions['Date'] >= today].copy()
        
        if len(upcoming) > 0:
            current_week = upcoming['Week'].min()
            week_games = predictions[predictions['Week'] == current_week].copy()
            
            st.subheader(f"Week {int(current_week)} Games ({len(week_games)})")
            week_games = week_games.sort_values('Date')
            for _, game in week_games.iterrows():
                display_game(game, standings, predictions, home_edge, ml_predictions)
        else:
            st.warning("No upcoming games found")
    
    elif page == "Custom Matchup":
        st.subheader("Custom Matchup")
        
        teams = sorted(standings['Team'].tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            away_team = st.selectbox("Away Team", ["Select..."] + teams)
        with col2:
            home_team = st.selectbox("Home Team", ["Select..."] + teams)
        
        if st.button("Generate Prediction"):
            if away_team != "Select..." and home_team != "Select..." and away_team != home_team:
                excel_pred = calculate_excel_prediction(home_team, away_team, standings, None, None, home_edge)
                
                # Try to find ML prediction
                ml_pred = None
                if ml_predictions is not None and len(ml_predictions) > 0:
                    ml_match = ml_predictions[
                        (ml_predictions['home_team'] == home_team) &
                        (ml_predictions['away_team'] == away_team)
                    ]
                    if len(ml_match) > 0:
                        ml_pred = get_ml_prediction(home_team, away_team, ml_match.iloc[0]['week'], ml_predictions)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Excel Model**")
                    st.write(f"Winner: {excel_pred['winner']}")
                    st.write(f"Score: {excel_pred['away_score']}-{excel_pred['home_score']}")
                    st.write(f"Confidence: {excel_pred['confidence']:.1%}")
                
                with col2:
                    st.markdown("**ML Model**")
                    if ml_pred:
                        st.write(f"Winner: {ml_pred['winner']}")
                        st.write(f"Score: {ml_pred['away_score']}-{ml_pred['home_score']}")
                        st.write(f"Confidence: {ml_pred['confidence']:.1%}")
                    else:
                        st.info("No ML prediction")
            else:
                st.error("Please select two different teams")
    
    elif page == "Performance":
        st.subheader("Model Performance")
        
        # Excel performance
        st.markdown("**Excel Model**")
        completed = predictions[predictions['Locked Correct'].isin(['YES', 'NO'])].copy()
        if len(completed) > 0:
            total = len(completed)
            correct = (completed['Locked Correct'] == 'YES').sum()
            accuracy = (correct / total * 100)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Games", total)
            col2.metric("Correct", correct)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
            
            # Week by week
            st.markdown("**Week by Week**")
            week_stats = completed.groupby('Week').apply(
                lambda x: pd.Series({
                    'Games': len(x),
                    'Correct': (x['Locked Correct'] == 'YES').sum(),
                    'Accuracy': f"{(x['Locked Correct'] == 'YES').sum() / len(x) * 100:.1f}%"
                })
            ).reset_index()
            st.dataframe(week_stats, hide_index=True, use_container_width=True)
        else:
            st.info("No completed games")
        
        st.divider()
        
        # ML performance
        st.markdown("**ML Model**")
        if ml_predictions is not None and len(ml_predictions) > 0:
            # Filter for completed games with ml_correct values
            ml_completed = ml_predictions[pd.notna(ml_predictions['ml_correct'])].copy()
            
            if len(ml_completed) > 0:
                ml_total = len(ml_completed)
                ml_correct = int((ml_completed['ml_correct'] == 1).sum())
                ml_accuracy = (ml_correct / ml_total * 100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Games", ml_total)
                col2.metric("Correct", ml_correct)
                col3.metric("Accuracy", f"{ml_accuracy:.1f}%")
                
                # Week by week
                if 'week' in ml_completed.columns:
                    st.markdown("**Week by Week**")
                    ml_week_stats = ml_completed.groupby('week').apply(
                        lambda x: pd.Series({
                            'Games': len(x),
                            'Correct': int((x['ml_correct'] == 1).sum()),
                            'Accuracy': f"{(x['ml_correct'] == 1).sum() / len(x) * 100:.1f}%"
                        })
                    ).reset_index()
                    ml_week_stats.columns = ['Week', 'Games', 'Correct', 'Accuracy']
                    st.dataframe(ml_week_stats, hide_index=True, use_container_width=True)
                
                # Model Comparison
                st.divider()
                st.markdown("**Model Comparison**")
                
                # Find games with both predictions
                excel_correct = int((ml_completed['excel_correct'] == 1).sum())
                excel_accuracy = (excel_correct / ml_total * 100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Excel Accuracy", f"{excel_accuracy:.1f}%", f"{excel_correct}/{ml_total}")
                col2.metric("ML Accuracy", f"{ml_accuracy:.1f}%", f"{ml_correct}/{ml_total}")
                diff = ml_accuracy - excel_accuracy
                col3.metric("Difference", f"{diff:+.1f}%")
                
                # Agreement stats
                both_correct = int(((ml_completed['ml_correct'] == 1) & (ml_completed['excel_correct'] == 1)).sum())
                both_wrong = int(((ml_completed['ml_correct'] == 0) & (ml_completed['excel_correct'] == 0)).sum())
                ml_only = int(((ml_completed['ml_correct'] == 1) & (ml_completed['excel_correct'] == 0)).sum())
                excel_only = int(((ml_completed['ml_correct'] == 0) & (ml_completed['excel_correct'] == 1)).sum())
                
                col1, col2 = st.columns(2)
                col1.metric("Both Correct", both_correct, f"{both_correct/ml_total*100:.1f}%")
                col2.metric("Both Wrong", both_wrong, f"{both_wrong/ml_total*100:.1f}%")
                
                col1, col2 = st.columns(2)
                col1.metric("ML Only Correct", ml_only)
                col2.metric("Excel Only Correct", excel_only)
                
            else:
                st.info("No completed games with ML predictions")
        else:
            st.info("ML model not available")

if __name__ == "__main__":
    main()
