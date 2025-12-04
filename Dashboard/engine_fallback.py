# engine_fallback.py
# -*- coding: utf-8 -*-
"""
Fallback implementations when Model_1_realtime_simulation is not available.
Provides demo data and minimal detector functionality for UI testing.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go


def load_tafe_data(path):
    """Generate demo data when real data is not available."""
    rng = pd.date_range("2025-03-01", "2025-05-31", freq="h")
    sites = ["Property 10001", "Property 11127", "Property 20002"]
    dfs = {}
    for s in sites:
        base = np.random.gamma(shape=2.0, scale=40.0, size=len(rng))
        base += np.where((rng.hour >= 0) & (rng.hour < 4), 80, 0)
        if s == "Property 11127":
            burst_mask = (rng >= "2025-04-10") & (rng <= "2025-04-20")
            base = base + np.where(burst_mask, 200, 0)
        df = pd.DataFrame(
            {
                "time": rng,
                "flow": np.clip(base + np.random.normal(0, 8, len(rng)), 0, None),
            }
        )
        dfs[s] = df
    return dfs


def validate_config(cfg):
    """Minimal config validation."""
    return True


def process_site(args):
    """Fallback process_site - returns None (not implemented in demo mode)."""
    return None, None, pd.DataFrame()


class SchoolLeakDetector:
    """Minimal detector implementation for demo mode."""

    def __init__(self, df, site_id, cfg, **kw):
        self.df = df.copy()
        self.site_id = site_id
        self.cfg = cfg
        self.daily = pd.DataFrame(
            index=pd.to_datetime(sorted(df["time"].dt.date.unique()))
        )
        if "flow" in df:
            nf = (
                df[df["time"].dt.hour.isin([0, 1, 2, 3])]
                .groupby(df["time"].dt.date)["flow"]
                .mean()
            )
            self.daily["NF_d"] = (
                pd.Series(nf.values, index=pd.to_datetime(nf.index))
                .reindex(self.daily.index)
                .fillna(method="ffill")
                .fillna(0)
            )
            ah = (
                df[(df["time"].dt.hour >= 16) | (df["time"].dt.hour < 7)]
                .groupby(df["time"].dt.date)["flow"]
                .sum()
            )
            self.daily["A_d"] = (
                pd.Series(ah.values, index=pd.to_datetime(ah.index))
                .reindex(self.daily.index)
                .fillna(method="ffill")
                .fillna(0)
                / 1000
            )
        else:
            self.daily["NF_d"] = 0
            self.daily["A_d"] = 0
        self.incidents = []

    def signals_and_score(self, d):
        """Return dummy signals and scores."""
        subs = {
            "MNF": 0.2,
            "RESIDUAL": 0.1,
            "CUSUM": 0.1,
            "AFTERHRS": 0.1,
            "BURSTBF": 0.0,
        }
        return subs, 30, 120, 10

    def to_plotly_figs(self, incident, window_days=30):
        """Return placeholder figures."""

        def ph(t):
            f = go.Figure()
            f.add_annotation(text=t, x=0.5, y=0.5, showarrow=False)
            f.update_layout(margin=dict(l=30, r=20, t=40, b=30))
            return f

        return (
            ph("No raw flow available"),
            ph("No NF trend"),
            ph("No After-hours"),
            ph("No weekly heatmap"),
        )
