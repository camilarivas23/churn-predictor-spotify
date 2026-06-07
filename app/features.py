"""
Feature Engineering Module
Generate 8-16 features across Recency, Frequency, Magnitude domains
and 4 feature types: Ratio, Time-based, Aggregation, Binary/Categorical
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureEngineer:
    """Generate features for churn prediction from listening events."""
    
    def __init__(self, events_df: pd.DataFrame, users_df: pd.DataFrame, reference_date: datetime = None):
        """Initialize feature engineer.
        
        Args:
            events_df: DataFrame with listening events (must have 'played_at' column)
            users_df: DataFrame with user profiles (must have user metadata)
            reference_date: Date to calculate recency from (default: today)
        """
        self.events_df = events_df.copy()
        self.users_df = users_df.copy()
        self.reference_date = reference_date or datetime.now()
        
        # Parse timestamps
        self.events_df['played_at'] = pd.to_datetime(self.events_df['played_at'])
        
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all 12 features across Recency, Frequency, Magnitude domains."""
        
        # Get unique users
        if 'user_id' not in self.events_df.columns:
            # Single user case
            user_data = {
                'user_id': self.users_df.iloc[0]['user_id'] if len(self.users_df) > 0 else 'unknown',
                'listening_events': len(self.events_df)
            }
            user_ids = [user_data['user_id']]
        else:
            user_ids = self.events_df['user_id'].unique()
        
        features_list = []
        
        for user_id in user_ids:
            if 'user_id' in self.events_df.columns:
                user_events = self.events_df[self.events_df['user_id'] == user_id]
            else:
                user_events = self.events_df
            
            if len(user_events) == 0:
                continue
            
            feature_dict = {'user_id': user_id}
            
            # ===== RECENCY FEATURES =====
            # 1. days_since_last_listen (Time-based, Recency) — Primary churn signal
            feature_dict['days_since_last_listen'] = self._days_since_last_listen(user_events)
            
            # 2. listening_events_last_7d (Aggregation, Recency)
            feature_dict['listening_events_last_7d'] = self._events_in_last_n_days(user_events, 7)
            
            # 3. avg_days_between_listens (Ratio, Recency) — Consistency of engagement
            feature_dict['avg_days_between_listens'] = self._avg_days_between_listens(user_events)
            
            # 4. is_active_last_7d (Binary/Categorical, Recency)
            feature_dict['is_active_last_7d'] = 1 if feature_dict['listening_events_last_7d'] > 0 else 0
            
            # ===== FREQUENCY FEATURES =====
            # 5. total_artist_listens (Aggregation, Frequency) — Total engagement depth
            feature_dict['total_artist_listens'] = len(user_events)
            
            # 6. unique_artists_90d (Aggregation, Frequency) — Genre diversity
            feature_dict['unique_artists_90d'] = self._unique_artists_in_last_n_days(user_events, 90)
            
            # 7. listen_to_artist_ratio (Ratio, Frequency) — Concentration of listening
            feature_dict['listen_to_artist_ratio'] = self._listen_concentration_ratio(user_events)
            
            # 8. repeat_listen_rate (Ratio, Frequency) — How often replays same artist
            feature_dict['repeat_listen_rate'] = self._repeat_listen_rate(user_events)
            
            # ===== MAGNITUDE FEATURES =====
            # 9. artist_popularity_avg (Aggregation, Magnitude) — Quality of consumed music
            feature_dict['artist_popularity_avg'] = self._avg_artist_popularity(user_events)
            
            # 10. track_energy_avg (Aggregation, Magnitude) — Engagement style (energy)
            feature_dict['track_energy_avg'] = self._avg_track_feature(user_events, 'energy')
            
            # 11. track_danceability_avg (Aggregation, Magnitude) — Engagement style (danceability)
            feature_dict['track_danceability_avg'] = self._avg_track_feature(user_events, 'danceability')
            
            # 12. is_high_engagement_user (Binary/Categorical, Magnitude) — High activity flag
            total_events = len(user_events)
            median_events = self.events_df.groupby('user_id' if 'user_id' in self.events_df.columns else pd.Series([0]*len(self.events_df))).size().median()
            feature_dict['is_high_engagement_user'] = 1 if total_events > median_events else 0
            
            # Additional time-based magnitude feature
            # 13. account_age_days (Time-based, Magnitude) — Tenure as stickiness proxy
            feature_dict['account_age_days'] = self._account_age_days(user_events)
            
            # Additional aggregation feature for balance
            # 14. avg_track_tempo (Aggregation, Magnitude)
            feature_dict['avg_track_tempo'] = self._avg_track_feature(user_events, 'tempo')
            
            features_list.append(feature_dict)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        return features_df
    
    def _days_since_last_listen(self, user_events: pd.DataFrame) -> float:
        """Days since most recent listening event."""
        if len(user_events) == 0:
            return np.nan
        last_listen = user_events['played_at'].max()
        days = (self.reference_date - last_listen).days
        return float(days)
    
    def _events_in_last_n_days(self, user_events: pd.DataFrame, n_days: int) -> int:
        """Count listening events in last n days."""
        cutoff_date = self.reference_date - timedelta(days=n_days)
        recent = user_events[user_events['played_at'] >= cutoff_date]
        return len(recent)
    
    def _avg_days_between_listens(self, user_events: pd.DataFrame) -> float:
        """Average days between consecutive listening events."""
        if len(user_events) < 2:
            return 0.0
        
        sorted_events = user_events.sort_values('played_at')
        timestamps = sorted_events['played_at'].values
        
        # Calculate days between consecutive events
        diffs = np.diff(pd.to_datetime(timestamps)).astype('timedelta64[D]').astype(float)
        
        if len(diffs) == 0:
            return 0.0
        
        return float(np.mean(diffs))
    
    def _unique_artists_in_last_n_days(self, user_events: pd.DataFrame, n_days: int) -> int:
        """Count unique artists listened in last n days."""
        cutoff_date = self.reference_date - timedelta(days=n_days)
        recent = user_events[user_events['played_at'] >= cutoff_date]
        
        unique_artists = set()
        for _, row in recent.iterrows():
            if 'artist_ids' in row and row['artist_ids']:
                if isinstance(row['artist_ids'], list):
                    unique_artists.update(row['artist_ids'])
                else:
                    unique_artists.add(row['artist_ids'])
        
        return len(unique_artists)
    
    def _listen_concentration_ratio(self, user_events: pd.DataFrame) -> float:
        """Ratio of top artist plays to total plays (concentration 0-1)."""
        if len(user_events) == 0:
            return 0.0
        
        # Count plays by artist
        artist_counts = {}
        for _, row in user_events.iterrows():
            if 'artist_ids' in row and row['artist_ids']:
                primary_artist = row['artist_ids'][0] if isinstance(row['artist_ids'], list) else row['artist_ids']
                artist_counts[primary_artist] = artist_counts.get(primary_artist, 0) + 1
        
        if not artist_counts:
            return 0.0
        
        top_artist_plays = max(artist_counts.values())
        total_plays = len(user_events)
        
        return float(top_artist_plays / total_plays)
    
    def _repeat_listen_rate(self, user_events: pd.DataFrame) -> float:
        """Proportion of repeat listens (same track within 30 min gaps excluded)."""
        if len(user_events) < 2:
            return 0.0
        
        sorted_events = user_events.sort_values('played_at')
        
        # Count sequential listens that are different
        different_tracks = 0
        prev_track = None
        
        for _, row in sorted_events.iterrows():
            track_id = row.get('track_id')
            if prev_track is not None and track_id != prev_track:
                different_tracks += 1
            prev_track = track_id
        
        total_transitions = len(sorted_events) - 1
        
        if total_transitions == 0:
            return 0.0
        
        # Repeat rate = 1 - (different tracks / transitions)
        return float(1.0 - (different_tracks / total_transitions))
    
    def _avg_artist_popularity(self, user_events: pd.DataFrame) -> float:
        """Average popularity of primary artists across listening events."""
        popularities = []
        
        for _, row in user_events.iterrows():
            if 'primary_artist_info' in row and isinstance(row['primary_artist_info'], dict):
                pop = row['primary_artist_info'].get('popularity')
                if pop is not None:
                    popularities.append(pop)
        
        if not popularities:
            return 0.0
        
        return float(np.mean(popularities))
    
    def _avg_track_feature(self, user_events: pd.DataFrame, feature_name: str) -> float:
        """Average audio feature (e.g., energy, danceability, tempo) across tracks."""
        features = []
        
        for _, row in user_events.iterrows():
            if 'track_features' in row and isinstance(row['track_features'], dict):
                value = row['track_features'].get(feature_name)
                if value is not None:
                    features.append(value)
        
        if not features:
            return 0.0
        
        return float(np.mean(features))
    
    def _account_age_days(self, user_events: pd.DataFrame) -> float:
        """Days since first listening event (proxy for account tenure)."""
        if len(user_events) == 0:
            return np.nan
        
        first_listen = user_events['played_at'].min()
        days = (self.reference_date - first_listen).days
        
        return float(days)
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature DataFrame."""
        features_df = features_df.fillna(0)
        return features_df


def create_features_from_raw_data(users_json_path: str, events_json_path: str, 
                                   artists_json_path: str = None, tracks_json_path: str = None) -> pd.DataFrame:
    """Load raw data from JSON files and generate features.
    
    Args:
        users_json_path: Path to users.json
        events_json_path: Path to listening_events.json
        artists_json_path: Path to artists.json (optional, for enrichment)
        tracks_json_path: Path to tracks.json (optional, for enrichment)
        
    Returns:
        Features DataFrame
    """
    import json
    
    # Load users
    with open(users_json_path, 'r') as f:
        users_data = json.load(f)
    users_df = pd.DataFrame(users_data)
    
    # Load events
    with open(events_json_path, 'r') as f:
        events_data = json.load(f)
    events_df = pd.DataFrame(events_data)
    
    # Engineer features
    engineer = FeatureEngineer(events_df, users_df)
    features_df = engineer.generate_all_features()
    
    return features_df


if __name__ == "__main__":
    # Example: Generate features from raw data
    features_df = create_features_from_raw_data(
        "data/raw/users.json",
        "data/raw/listening_events.json"
    )
    print(features_df)
    print(f"\nGenerated {len(features_df.columns)} features")
    print(f"Feature columns: {features_df.columns.tolist()}")
