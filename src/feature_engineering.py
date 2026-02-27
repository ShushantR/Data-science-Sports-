import pandas as pd
import numpy as np

def create_match_features(df_matches, df_balls):
    """
    Transform raw matches and ball-by-ball data into a training dataset for win prediction.
    
    Mathematical Features Created:
    - runs_left: target - current_score
    - balls_left: 120 - current_delivery_count
    - wickets_left: 10 - cumulative_wickets
    - CRR (Current Run Rate): (current_score * 6) / balls_bowled
    - RRR (Required Run Rate): (runs_left * 6) / balls_left
    """
    # 1. Calculate target score from 1st innings
    total_score_df = df_balls[df_balls['inning'] == 1].groupby('id')[['total_runs']].sum().reset_index()
    total_score_df.rename(columns={'total_runs': 'target_score'}, inplace=True)
    
    # 2. Merge target score back to match context
    match_df = df_matches.merge(total_score_df, on='id')
    match_df['target_score'] = match_df['target_score'] + 1  # Target is 1st inn runs + 1
    
    # 3. Filter for 2nd innings deliveries
    delivery_df = df_balls[df_balls['inning'] == 2].copy()
    
    # 4. Map target score and winner to deliveries
    delivery_df = delivery_df.merge(match_df[['id', 'target_score', 'winner', 'city']], on='id')
    
    # 5. Cumulative match state
    delivery_df['current_score'] = delivery_df.groupby('id')['total_runs'].cumsum()
    delivery_df['runs_left'] = delivery_df['target_score'] - delivery_df['current_score']
    
    # Calculate balls left (handles 6 balls/over)
    delivery_df['balls_bowled'] = (delivery_df['over'] * 6) + delivery_df['ball']
    delivery_df['balls_left'] = 120 - delivery_df['balls_bowled']
    
    # Calculate wickets left
    delivery_df['wickets_left'] = 10 - delivery_df.groupby('id')['is_wicket'].cumsum()
    
    # 6. Performance Metrics
    # Avoid division by zero for CRR/RRR
    delivery_df['crr'] = np.where(delivery_df['balls_bowled'] == 0, 0, (delivery_df['current_score'] * 6) / delivery_df['balls_bowled'])
    delivery_df['rrr'] = np.where(delivery_df['balls_left'] <= 0, 0, (delivery_df['runs_left'] * 6) / delivery_df['balls_left'])
    
    # 7. Final categorical/target encoding
    delivery_df['result'] = (delivery_df['batting_team'] == delivery_df['winner']).astype(int)
    
    # Pressure Index: A custom metric normalized to situational leverage
    delivery_df['pressure_index'] = np.where(delivery_df['crr'] == 0, delivery_df['rrr'], delivery_df['rrr'] / delivery_df['crr'])
    
    features = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 
                'wickets_left', 'target_score', 'crr', 'rrr', 'pressure_index', 'result']
    
    return delivery_df[features].dropna()
