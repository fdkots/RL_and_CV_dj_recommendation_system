import numpy as np
import pandas as pd

#README 
def target_excitement(p: float) -> float:
    """
    Typical DJ set arc:
    - Warm up slowly (0 → 0.3)
    - Build to peak (0.3 → 0.7)
    - Sustain peak (0.7 → 0.85)
    - Brief dip and final climax (0.85 → 1.0)
    """
    return (
        0.3 * np.sin(np.pi * p)          # base sinusoidal arc
        + 0.5 * p                          # linear energy build
        + 0.2 * np.sin(3 * np.pi * p)    # sub-peaks for variation
    )
        
class DJEnvironment:
    FEATURE_COLS = ["bpm", "energy", "danceability"]

    def __init__(self, track_df: pd.DataFrame, result_df: pd.DataFrame, max_steps: int = 20):
        # Merge sentiment reward into tracks
        sentiment = result_df[['song', 'reward', 'pct_Positive', 'pct_Neutral', 'pct_Negative']]
        merged = track_df.merge(sentiment, on='song', how='left')
        merged['reward'] = merged['reward'].fillna(0.0)  # songs with no crowd data = 0 reward
        
        self.tracks    = merged[["song", "dj"] + self.FEATURE_COLS + ['reward', 'pct_Positive', 'pct_Negative']].dropna(subset=self.FEATURE_COLS).reset_index(drop=True)
        self.max_steps = max_steps
        self._normalize()
        self.current_idx = 0
        self.step_count  = 0
        self.played      = set()

    def _normalize(self):
        for col in self.FEATURE_COLS:
            lo, hi = self.tracks[col].min(), self.tracks[col].max()
            self.tracks[f"{col}_norm"] = (self.tracks[col] - lo) / (hi - lo + 1e-8)
        self.norm_cols = [f"{c}_norm" for c in self.FEATURE_COLS]

    def _state(self):
        return self.tracks.loc[self.current_idx, self.norm_cols].values.astype(np.float32)

    def reset(self):
        self.current_idx = np.random.randint(len(self.tracks))
        self.step_count  = 0
        self.played      = {self.current_idx}
        return self._state()

    def step(self, action: int):
        current  = self.tracks.iloc[self.current_idx]
        nxt      = self.tracks.iloc[action]
        progress = self.step_count / self.max_steps

        # --- 1. Audio Flow Rewards (The DJ Rules) ---
        # The agent needs to follow the classic "DJ Energy Curve" (warmup -> peak -> cooldown)
        ideal_energy = np.exp(-0.5 * ((progress - 0.7) / 0.2) ** 2)
        r_energy = np.exp(-5.0 * (nxt["energy_norm"] - ideal_energy) ** 2)
        
        # Don't trainwreck the tempo! 
        r_bpm_jump = -abs(nxt["bpm"] - current["bpm"]) / 60.0
        
        # Base danceability of the track itself
        r_dance = nxt["danceability_norm"]

        # --- 2. Multi-Modal Vision Rewards (The Crowd Feedback!) ---
        
        # Vision Metric 1: Facial Sentiment (from EMONEXT/DeepFace)
        # Did the crowd look happy or bored during this track? Range: [-1, 1]
        r_sentiment = nxt["pct_Positive"] - nxt["pct_Negative"]
        
        # Vision Metric 2: Physical Dancing (from Lucas-Kanade Optical Flow)
        # Normalize motion so massive spikes don't break the Q-learning targets
        # Assuming you calculated 'avg_motion_energy' in the previous step
        r_motion = nxt.get("avg_motion_energy", 0)  
        
        # (Optional) If motion absolute values are huge (e.g. 50.0), scale them down:
        r_motion_scaled = min(1.0, r_motion / 10.0) 

        # --- Final Weighted Reward Combination ---
        # As the night goes on, crowd reaction matters MORE than the audio rules!
        w_energy    = 1.0
        w_bpm       = 0.5
        w_dance     = 0.5
        w_sentiment = 1.0  # Crucial for warmup/cooldown
        w_motion    = 2.0  # The ultimate goal of a DJ: Make them move!

        reward = (w_energy * r_energy) + \
                 (w_bpm * r_bpm_jump) + \
                 (w_dance * r_dance) + \
                 (w_sentiment * r_sentiment) + \
                 (w_motion * r_motion_scaled)

        self.played.add(action)
        self.current_idx = action
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._state(), reward, done


    @property
    def n_actions(self): return len(self.tracks)

    @property
    def state_size(self): return len(self.norm_cols)
