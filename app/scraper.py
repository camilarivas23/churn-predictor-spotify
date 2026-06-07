"""
Spotify OAuth + API Data Collection Module
Handles secure OAuth flow, token refresh, and rate-limited API calls.
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class SpotifyDataCollector:
    """Handles Spotify API data collection with OAuth authentication and rate limiting."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, data_dir: str = "./data/raw"):
        """Initialize Spotify OAuth collector.
        
        Args:
            client_id: Spotify app client ID
            client_secret: Spotify app client secret
            redirect_uri: OAuth redirect URI
            data_dir: Directory to store raw data
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize OAuth
        self.auth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-read-recently-played user-read-private user-follow-read"
        )
        self.sp = spotipy.Spotify(auth_manager=self.auth)
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # seconds
        self.last_call = 0
        
        # Request session with retries
        self.session = self._create_session()
        
        # Data storage
        self.users_data = []
        self.listening_events = []
        self.artists_cache = {}
        self.tracks_cache = {}
    
    def _create_session(self) -> requests.Session:
        """Create requests session with exponential backoff for rate limiting."""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _respect_rate_limit(self):
        """Respect API rate limits with minimal delay between calls."""
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call = time.time()
    
    def fetch_user_profile(self) -> Dict:
        """Fetch authenticated user's profile.
        
        Returns:
            User profile data: id, display_name, followers, external_urls, created_at (inferred)
        """
        self._respect_rate_limit()
        try:
            user = self.sp.current_user()
            return {
                "user_id": user.get("id"),
                "display_name": user.get("display_name"),
                "followers": user.get("followers", {}).get("total", 0),
                "external_urls": user.get("external_urls", {}),
                "fetched_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            return {}
    
    def fetch_recent_tracks(self, limit: int = 50, paged: bool = True) -> List[Dict]:
        """Fetch user's recently played tracks.
        
        Args:
            limit: Max tracks per page (Spotify max: 50)
            paged: If True, fetch all available pages; else just one
            
        Returns:
            List of track play events with timestamps
        """
        self._respect_rate_limit()
        all_tracks = []
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            all_tracks.extend(results.get("items", []))
            
            # Paginate if requested
            while paged and results.get("next"):
                self._respect_rate_limit()
                results = self.sp.next(results)
                all_tracks.extend(results.get("items", []))
            
            # Parse track events
            events = []
            for item in all_tracks:
                track = item.get("track", {})
                artists = track.get("artists", [])
                event = {
                    "track_id": track.get("id"),
                    "track_name": track.get("name"),
                    "artist_ids": [a.get("id") for a in artists],
                    "artist_names": [a.get("name") for a in artists],
                    "played_at": item.get("played_at"),
                    "fetched_at": datetime.now().isoformat()
                }
                events.append(event)
            
            return events
        except Exception as e:
            print(f"Error fetching recent tracks: {e}")
            return []
    
    def fetch_artist_info(self, artist_id: str) -> Dict:
        """Fetch artist metadata (popularity, followers, genres).
        
        Args:
            artist_id: Spotify artist ID
            
        Returns:
            Artist metadata dict
        """
        # Check cache first
        if artist_id in self.artists_cache:
            return self.artists_cache[artist_id]
        
        self._respect_rate_limit()
        try:
            artist = self.sp.artist(artist_id)
            data = {
                "artist_id": artist.get("id"),
                "name": artist.get("name"),
                "popularity": artist.get("popularity", 0),
                "followers": artist.get("followers", {}).get("total", 0),
                "genres": artist.get("genres", []),
                "external_urls": artist.get("external_urls", {}),
                "fetched_at": datetime.now().isoformat()
            }
            self.artists_cache[artist_id] = data
            return data
        except Exception as e:
            print(f"Error fetching artist {artist_id}: {e}")
            return {}
    
    def fetch_track_features(self, track_id: str) -> Dict:
        """Fetch audio features for a track (energy, danceability, tempo, etc).
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Audio features dict
        """
        # Check cache first
        if track_id in self.tracks_cache:
            return self.tracks_cache[track_id]
        
        self._respect_rate_limit()
        try:
            features = self.sp.audio_features(track_id)[0]
            if features:
                data = {
                    "track_id": track_id,
                    "energy": features.get("energy"),
                    "danceability": features.get("danceability"),
                    "tempo": features.get("tempo"),
                    "valence": features.get("valence"),
                    "acousticness": features.get("acousticness"),
                    "instrumentalness": features.get("instrumentalness"),
                    "liveness": features.get("liveness"),
                    "loudness": features.get("loudness"),
                    "speechiness": features.get("speechiness"),
                    "fetched_at": datetime.now().isoformat()
                }
                self.tracks_cache[track_id] = data
                return data
        except Exception as e:
            print(f"Error fetching track features for {track_id}: {e}")
        
        return {}
    
    def collect_user_cohort(self) -> tuple:
        """Orchestrate data collection for current user.
        
        Returns:
            Tuple of (users_df, events_df, artists_df, tracks_df)
        """
        print("Starting data collection...")
        
        # 1. Fetch user profile
        print("Fetching user profile...")
        user_profile = self.fetch_user_profile()
        if not user_profile:
            print("Failed to fetch user profile. Exiting.")
            return None, None, None, None
        
        self.users_data.append(user_profile)
        user_id = user_profile["user_id"]
        
        # 2. Fetch recent tracks
        print(f"Fetching recent tracks for user {user_id}...")
        recent_tracks = self.fetch_recent_tracks(limit=50, paged=True)
        print(f"Fetched {len(recent_tracks)} track events")
        
        # Enrich with artist and track features
        print("Enriching track events with artist and audio features...")
        enriched_events = []
        for i, event in enumerate(recent_tracks):
            if i % 10 == 0:
                print(f"  Processing event {i+1}/{len(recent_tracks)}...")
            
            track_id = event["track_id"]
            artist_ids = event["artist_ids"]
            
            # Fetch track audio features
            track_features = self.fetch_track_features(track_id)
            event.update({
                "track_features": track_features
            })
            
            # Fetch artist info for first artist (primary artist)
            if artist_ids:
                artist_info = self.fetch_artist_info(artist_ids[0])
                event.update({
                    "primary_artist_info": artist_info
                })
            
            enriched_events.append(event)
        
        self.listening_events.extend(enriched_events)
        
        # 3. Save raw data
        print("Saving raw data to disk...")
        self._save_raw_data()
        
        # 4. Create DataFrames
        print("Creating DataFrames...")
        users_df = pd.DataFrame(self.users_data)
        events_df = pd.DataFrame(self.listening_events)
        
        print(f"\nData collection complete!")
        print(f"  Users: {len(users_df)}")
        print(f"  Listening events: {len(events_df)}")
        print(f"  Unique artists cached: {len(self.artists_cache)}")
        print(f"  Unique tracks cached: {len(self.tracks_cache)}")
        
        return users_df, events_df, pd.DataFrame(list(self.artists_cache.values())), pd.DataFrame(list(self.tracks_cache.values()))
    
    def _save_raw_data(self):
        """Save raw data as JSON files for later processing."""
        # Save users
        if self.users_data:
            users_path = os.path.join(self.data_dir, "users.json")
            with open(users_path, 'w') as f:
                json.dump(self.users_data, f, indent=2)
            print(f"Saved users data to {users_path}")
        
        # Save listening events
        if self.listening_events:
            events_path = os.path.join(self.data_dir, "listening_events.json")
            with open(events_path, 'w') as f:
                json.dump(self.listening_events, f, indent=2)
            print(f"Saved listening events to {events_path}")
        
        # Save artists cache
        if self.artists_cache:
            artists_path = os.path.join(self.data_dir, "artists.json")
            with open(artists_path, 'w') as f:
                json.dump(list(self.artists_cache.values()), f, indent=2)
            print(f"Saved artists data to {artists_path}")
        
        # Save tracks cache
        if self.tracks_cache:
            tracks_path = os.path.join(self.data_dir, "tracks.json")
            with open(tracks_path, 'w') as f:
                json.dump(list(self.tracks_cache.values()), f, indent=2)
            print(f"Saved tracks data to {tracks_path}")


def main():
    """Test data collection with OAuth credentials."""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
    
    if not client_id or not client_secret:
        print("Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET not set in .env")
        return
    
    collector = SpotifyDataCollector(client_id, client_secret, redirect_uri)
    users_df, events_df, artists_df, tracks_df = collector.collect_user_cohort()


if __name__ == "__main__":
    main()
