# Updated Model_1.py with modifications for up_to_date

# %%
import time
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
import logging
import os
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor  # Added for parallel processing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("leak_detection.log"), logging.StreamHandler()],
)


class SchoolLeakDetector:
    def __init__(self, df, site_id, cfg, leak_log=None, up_to_date=None):
        self.df = df.copy()
        self.site_id = site_id
        self.cfg = cfg
        self.leak_log = leak_log if leak_log else []
        self.incidents = []
        self.excluded_dates = set()
        self.daily = None
        self.pattern_suppressions = {}
        self.theta_min = None  # adaptive threshold (MNF baseline)
        self.up_to_date = pd.to_datetime(up_to_date) if up_to_date else None
        # ✅ FIX: Store signal components by date to enable consistent confidence calculation
        # Even when rolling baselines change, we use the originally calculated signals
        self.signal_components_by_date = {}  # {date: {sub_scores, deltaNF, NF_MAD}}
        # ✅ FIX: Store FROZEN confidence values to prevent recalculation
        # Once a confidence is calculated for a date, it NEVER changes
        self.confidence_by_date = {}  # {date_key: confidence_value}

    def preprocess(self):
        # Ensure timezone-naive timestamps and hourly alignment
        self.df["time"] = pd.to_datetime(self.df["time"]).dt.tz_localize(None)
        if self.up_to_date:
            self.df = self.df[self.df["time"] <= self.up_to_date]
        self.df.set_index("time", inplace=True)
        self.df = self.df.resample("h").sum()
        self.df.interpolate(method="linear", limit=3, inplace=True)
        self.df["flow"].fillna(0, inplace=True)
        self.df["flow"] = self.df["flow"].clip(lower=0)
        self.df["date"] = self.df.index.date
        self.df["hour"] = self.df.index.hour
        q99_9 = self.df["flow"].quantile(0.999)
        self.df["outlier"] = self.df["flow"] > q99_9
        logging.info(f"{self.site_id}: Preprocessed {len(self.df)} hourly records")

    def robust_median(self, series):
        return np.median(series) if len(series) > 0 else 0

    def robust_mad(self, series):
        med = self.robust_median(series)
        return np.median(np.abs(series - med)) if len(series) > 0 else 0

    def detect_cusum(self, series, k, h, mad):
        mean = np.mean(series)
        s_plus = np.zeros(len(series))
        for i in range(1, len(series)):
            s_plus[i] = max(0, s_plus[i - 1] + (series[i] - mean - k * mad))
        return 1 if np.any(s_plus > h * mad) else 0

    def get_rolling_baseline(self, d, metric, daily_series):
        d = pd.to_datetime(d).date()
        hist_start = d - timedelta(days=self.cfg["baseline_window_days"])
        sub = daily_series[
            (daily_series.index.date >= hist_start) & (daily_series.index.date < d)
        ]
        sub = sub[~sub.index.isin([pd.Timestamp(date) for date in self.excluded_dates])]
        return self.robust_median(sub), self.robust_mad(sub)

    def get_hourly_profile(self, h, d):
        d = pd.to_datetime(d).date()
        hist_start = d - timedelta(days=self.cfg["baseline_window_days"])
        mask = (
            (self.df["date"] >= hist_start)
            & (self.df["date"] < d)
            & (self.df["hour"] == h)
        )
        sub = self.df[mask]["flow"]
        sub = sub[
            ~pd.Series(self.df[mask].index.date, index=self.df[mask].index).isin(
                self.excluded_dates
            )
        ]
        return self.robust_median(sub), self.robust_mad(sub)

    def baselining(self):
        night_mask = (self.df["hour"] >= self.cfg["night_start"]) & (
            self.df["hour"] < self.cfg["night_end"]
        )
        self.daily = pd.DataFrame(index=pd.to_datetime(np.unique(self.df["date"])))
        self.daily["NF_d"] = (
            self.df[night_mask]
            .groupby("date")["flow"]
            .apply(lambda x: np.percentile(x, 10))
        )

        # ✅ Correct after-hours: hour >= after_hours_start OR hour < after_hours_end
        ah_mask = (self.df["hour"] >= self.cfg["after_hours_start"]) | (
            self.df["hour"] < self.cfg["after_hours_end"]
        )
        self.daily["A_d"] = (
            self.df[ah_mask].groupby("date")["flow"].sum() / 1000.0
        )  # kL

        logging.info(f"{self.site_id}: Baselined {len(self.daily)} days")

    def signals_and_score(self, d):
        d_dt = pd.to_datetime(d)
        NF_d = self.daily.loc[d, "NF_d"]
        NF_base, NF_MAD = self.get_rolling_baseline(d, "NF_d", self.daily["NF_d"])
        deltaNF = max(0, NF_d - NF_base)
        thresh = max(3 * NF_MAD, self.cfg["abs_floor_lph"])
        s_MNF = (
            max(0, min(1, (deltaNF - thresh) / (thresh + self.cfg["abs_floor_lph"])))
            if deltaNF > thresh
            else 0
        )

        # Vectorized residual calculation
        d_date = d_dt.date()
        after_hours_data = self.df[
            (self.df["date"] == d_date)
            & (
                (self.df["hour"] >= self.cfg["after_hours_start"])
                | (self.df["hour"] < self.cfg["after_hours_end"])
            )
        ]
        residuals = []
        mad_rs = []
        if not after_hours_data.empty:
            hourly_profiles = [
                self.get_hourly_profile(h, d) for h in after_hours_data["hour"]
            ]
            residuals = after_hours_data["flow"] - pd.Series(
                [p[0] for p in hourly_profiles], index=after_hours_data.index
            )
            mad_rs = [p[1] for p in hourly_profiles]
        after_hours_count = len(residuals)
        s_RES = 0
        if after_hours_count > 0:
            med_res = np.median(residuals)
            med_mad_r = np.median(mad_rs)
            pos_frac = sum(r > 0 for r in residuals) / after_hours_count
            thresh_r = max(3 * med_mad_r, self.cfg["abs_floor_lph"])
            s_RES = (
                1
                if pos_frac >= 0.7 and med_res > thresh_r
                else max(0, min(1, med_res / (2 * thresh_r)))
            )

        hist_NF = self.daily["NF_d"][:d]
        mad_nf_hist = self.robust_mad(hist_NF)
        cusum_NF = self.detect_cusum(
            hist_NF.values, self.cfg["cusum_k"], self.cfg["cusum_h"], mad_nf_hist
        )
        hist_A = self.daily["A_d"][:d]
        mad_a_hist = self.robust_mad(hist_A)
        cusum_A = self.detect_cusum(
            hist_A.values, self.cfg["cusum_k"], self.cfg["cusum_h"], mad_a_hist
        )
        s_CUSUM = max(cusum_NF, cusum_A)

        A_d = self.daily.loc[d, "A_d"]
        A_base, A_MAD = self.get_rolling_baseline(d, "A_d", self.daily["A_d"])
        deltaA = A_d - A_base
        threshA = max(3 * A_MAD, self.cfg["sustained_after_hours_delta_kl"])
        prev_d = d - timedelta(days=1)
        s_AH = 0
        if prev_d in self.daily.index:
            prev_deltaA = (
                self.daily.loc[prev_d, "A_d"]
                - self.get_rolling_baseline(prev_d, "A_d", self.daily["A_d"])[0]
            )
            sustained = deltaA > threshA and prev_deltaA > threshA
            s_AH = 1 if sustained else max(0, min(1, deltaA / (2 * threshA)))

        s_BF = 0
        daily_data = self.df[(self.df["date"] == d_date)]
        if not daily_data.empty:
            hourly_profiles = [
                self.get_hourly_profile(h, d)[0] for h in daily_data["hour"]
            ]
            spikes = (
                daily_data["flow"]
                > pd.Series(hourly_profiles, index=daily_data.index)
                * self.cfg["spike_multiplier"]
            )
            if spikes.any():
                next_d = d + timedelta(days=1)
                if next_d in self.daily.index:
                    next_deltaNF = (
                        self.daily.loc[next_d, "NF_d"]
                        - self.get_rolling_baseline(next_d, "NF_d", self.daily["NF_d"])[
                            0
                        ]
                    )
                    if next_deltaNF > max(
                        3
                        * self.get_rolling_baseline(next_d, "NF_d", self.daily["NF_d"])[
                            1
                        ],
                        self.cfg["abs_floor_lph"],
                    ):
                        s_BF = 1

        sub_scores = {
            "MNF": s_MNF,
            "RESIDUAL": s_RES,
            "CUSUM": s_CUSUM,
            "AFTERHRS": s_AH,
            "BURSTBF": s_BF,
        }
        leak_score = (
            sum(self.cfg["score_weights"][k] * v for k, v in sub_scores.items()) * 100
        )
        leak_score = min(100, max(0, leak_score))

        return sub_scores, leak_score, deltaNF, NF_MAD

    def diagnose_burstbf(self, d):
        """Diagnostic method to check BURST/BF signal calculation for a specific date"""
        d_date = d.date() if hasattr(d, "date") else d

        # Get the daily data for this date
        daily_data = self.df[(self.df["date"] == d_date)]

        if daily_data.empty:
            return {
                "error": "No data found for this date",
                "date": str(d_date),
            }

        # Check for spikes
        hourly_profiles = []
        spike_thresholds = []
        actual_flows = []
        spikes_detected = []

        for idx, row in daily_data.iterrows():
            h = row["hour"]
            profile_val, _ = self.get_hourly_profile(h, d)
            threshold = profile_val * self.cfg["spike_multiplier"]
            is_spike = row["flow"] > threshold

            hourly_profiles.append(profile_val)
            spike_thresholds.append(threshold)
            actual_flows.append(row["flow"])
            spikes_detected.append(is_spike)

        has_spikes = any(spikes_detected)

        # Check next day's NF increase if spikes detected
        next_day_check = None
        s_BF_value = 0

        if has_spikes:
            next_d = d + timedelta(days=1)
            if next_d in self.daily.index:
                baseline_nf, mad_nf = self.get_rolling_baseline(
                    next_d, "NF_d", self.daily["NF_d"]
                )
                next_nf = self.daily.loc[next_d, "NF_d"]
                next_deltaNF = next_nf - baseline_nf
                required_threshold = max(3 * mad_nf, self.cfg["abs_floor_lph"])

                next_day_check = {
                    "next_date": str(next_d.date()),
                    "next_day_NF": float(next_nf),
                    "baseline_NF": float(baseline_nf),
                    "delta_NF": float(next_deltaNF),
                    "MAD": float(mad_nf),
                    "required_threshold": float(required_threshold),
                    "threshold_met": next_deltaNF > required_threshold,
                }

                if next_deltaNF > required_threshold:
                    s_BF_value = 1
            else:
                next_day_check = {"error": "Next day not in data"}

        return {
            "date": str(d_date),
            "has_spikes_detected": has_spikes,
            "spike_count": sum(spikes_detected),
            "spike_multiplier_config": self.cfg["spike_multiplier"],
            "hourly_details": [
                {
                    "hour": int(daily_data.iloc[i]["hour"]),
                    "actual_flow": float(actual_flows[i]),
                    "hourly_profile": float(hourly_profiles[i]),
                    "spike_threshold": float(spike_thresholds[i]),
                    "is_spike": bool(spikes_detected[i]),
                    "exceeds_by": float(actual_flows[i] - spike_thresholds[i]),
                }
                for i in range(len(actual_flows))
            ],
            "next_day_check": next_day_check,
            "final_BURSTBF_score": s_BF_value,
        }

    def get_severity(self, deltaNF):
        for s, (low, high) in self.cfg["severity_bands_lph"].items():
            if low <= deltaNF < high:
                return s
        return "S5" if deltaNF >= 10000 else "S1"

    def to_dashboard_dict(self, incident):
        """Return incident dict with serializable types for dashboard UI"""
        return {
            "site_id": incident["site_id"],
            "status": incident.get("status", ""),
            "start_day": pd.to_datetime(incident["start_day"]),
            "last_day": pd.to_datetime(incident["last_day"]),
            "severity_max": incident.get("severity_max", "S1"),
            "confidence": float(incident.get("confidence", 0)),
            "volume_lost_kL": float(incident.get("volume_lost_kL", 0)),
            "reason_codes": list(incident.get("reason_codes", [])),
            "alert_date": pd.to_datetime(
                incident.get("alert_date", incident["last_day"])
            ),
        }

    def get_confidence(self, sub_scores, persistence_days, deltaNF, NF_MAD):
        sig_agree = sum(1 for v in sub_scores.values() if v >= 0.7)
        snr = deltaNF / max(NF_MAD, 1)
        norm_snr = min(1, snr / 10)
        norm_persist = min(1, persistence_days / 10)
        norm_agree = sig_agree / 5
        confidence = (0.3 * norm_snr + 0.3 * norm_persist + 0.4 * norm_agree) * 100
        return min(100, max(0, confidence))

    def create_confidence_evolution_mini(self, incident):
        """Create mini bar chart showing confidence building over duration period"""
        import plotly.graph_objects as go

        # ✅ FIX: Use stored signal components to calculate confidence consistently
        # This prevents recalculation when events are extended
        dates = []
        confidences = []

        event_id = incident.get("event_id", "unknown")
        logging.info(f"[CHART] Creating confidence evolution for {event_id}")

        # PREFERRED: Use stored signal components from incident
        if (
            "signal_components_by_date" in incident
            and incident["signal_components_by_date"]
        ):
            start = pd.to_datetime(incident["start_day"])
            end = pd.to_datetime(incident["last_day"])

            logging.info(
                f"[CHART] {event_id}: Using STORED signal components for {start.date()} to {end.date()}"
            )
            logging.info(
                f"[CHART] {event_id}: Available dates in signal_components: {list(incident['signal_components_by_date'].keys())}"
            )

            try:
                for d in pd.date_range(start, end, freq="D"):
                    d_key = d.strftime("%Y-%m-%d")
                    if d_key in incident["signal_components_by_date"]:
                        comp = incident["signal_components_by_date"][d_key]

                        # ✅ FIX: Use FROZEN confidence if available, otherwise calculate
                        if "confidence" in comp:
                            conf = comp["confidence"]
                            logging.info(
                                f"[CHART] {event_id} - {d_key}: Using FROZEN confidence={conf:.1f}%"
                            )
                        else:
                            # Legacy fallback: calculate confidence with the RELATIVE persistence from start
                            persistence_days = (d - start).days + 1
                            conf = self.get_confidence(
                                comp["sub_scores"],
                                persistence_days,
                                comp["deltaNF"],
                                comp["NF_MAD"],
                            )
                            logging.warning(
                                f"[CHART] {event_id} - {d_key}: No frozen confidence, recalculating (may be inconsistent)"
                            )

                        dates.append(d)
                        confidences.append(conf)

                        logging.info(
                            f"[CHART] {event_id} - {d_key}: deltaNF={comp['deltaNF']:.1f}, "
                            f"NF_MAD={comp['NF_MAD']:.1f}, conf={conf:.1f}%"
                        )
            except Exception as e:
                logging.error(f"[CHART] {event_id}: Error using signal components: {e}")
                pass

        # Fallback: Use legacy confidence_evolution_daily if available
        if not dates or not confidences:
            logging.warning(
                f"[CHART] {event_id}: No signal components found, trying legacy confidence_evolution_daily"
            )
            if (
                "confidence_evolution_daily" in incident
                and incident["confidence_evolution_daily"]
            ):
                try:
                    for entry in incident["confidence_evolution_daily"]:
                        dates.append(pd.to_datetime(entry["date"]))
                        confidences.append(entry["confidence"])
                    logging.info(
                        f"[CHART] {event_id}: Using legacy confidence_evolution_daily"
                    )
                except Exception as e:
                    logging.error(f"[CHART] {event_id}: Error using legacy data: {e}")
                    pass

        # Final fallback: recalculate from scratch (DEPRECATED - will be inconsistent)
        if not dates or not confidences:
            logging.warning(
                f"[CHART] {event_id}: RECALCULATING from scratch (INCONSISTENT!)"
            )
            start = pd.to_datetime(incident["start_day"])
            end = pd.to_datetime(incident["last_day"])

            for i, d in enumerate(pd.date_range(start, end, freq="D")):
                if d not in self.daily.index:
                    continue

                try:
                    sub_scores, _, deltaNF, NF_MAD = self.signals_and_score(d)
                    persistence_days = (d - start).days + 1
                    conf = self.get_confidence(
                        sub_scores, persistence_days, deltaNF, NF_MAD
                    )

                    dates.append(d)
                    confidences.append(conf)

                    logging.warning(
                        f"[CHART] {event_id} - {d.strftime('%Y-%m-%d')}: RECALC persist={persistence_days}, "
                        f"deltaNF={deltaNF:.1f}, conf={conf:.1f}%"
                    )
                except Exception:
                    continue

        # Create figure
        fig = go.Figure()

        # Log the final confidence values that will be plotted
        if dates and confidences:
            conf_summary = ", ".join(
                [f"{d.strftime('%m-%d')}:{c:.0f}%" for d, c in zip(dates, confidences)]
            )
            logging.info(
                f"[CHART] {event_id}: FINAL confidence values: [{conf_summary}]"
            )

        if dates and confidences:
            # Color bars based on confidence level
            bar_colors = [
                (
                    "rgb(255,0,0)"
                    if c < 50
                    else "rgb(255,165,0)" if c < 70 else "rgb(0,255,0)"
                )
                for c in confidences
            ]

            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=confidences,
                    marker=dict(color=bar_colors, line=dict(width=1, color="white")),
                    text=[f"{c:.0f}%" for c in confidences],
                    textposition="outside",
                    textfont=dict(size=10, color="white"),
                    hovertemplate="<b>Day %{x|%b %d}</b><br>Confidence: %{y:.1f}%<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis=dict(
                title="",
                showgrid=False,
                tickformat="%b %d",
            ),
            yaxis=dict(
                title="Confidence %",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                range=[0, 110],  # Extra space for text labels on top
            ),
            template="plotly_dark",
            height=200,
            margin=dict(l=50, r=20, t=20, b=30),
            paper_bgcolor="rgba(30,30,30,1)",
            plot_bgcolor="rgba(20,20,20,1)",
            showlegend=False,
        )

        return fig

    def get_persistence_needed(self, deltaNF, sig_agree, confidence):
        if deltaNF < 100:
            gate = self.cfg["persistence_gates"]["<100"]
        elif deltaNF < 200:
            gate = self.cfg["persistence_gates"]["100-200"]
        elif deltaNF < 1000:
            gate = self.cfg["persistence_gates"]["200-1000"]
        else:
            gate = self.cfg["persistence_gates"][">=1000"]

        needed = (
            gate["fast_min"]
            if sig_agree >= 3 and confidence >= 70
            else gate["default_max"]
        )
        return max(3, needed)  # <-- Enforce minimum 3 days

    def get_adaptive_threshold(self):
        night_mask = (self.df["hour"] >= self.cfg["night_start"]) & (
            self.df["hour"] < self.cfg["night_end"]
        )
        mnf = (
            self.df[night_mask]
            .groupby("date")["flow"]
            .apply(lambda x: np.percentile(x, 10))
        )
        self.theta_min = max(
            self.cfg["abs_floor_lph"],
            self.robust_median(mnf) + 2 * self.robust_mad(mnf),
        )

        logging.info(f"{self.site_id}: Adaptive theta_min = {self.theta_min} L/h")
        return {"theta_min": self.theta_min}

    def categorize_leak(self, incident):
        # Use site-specific MNF baseline for scaling
        baseline = self.theta_min if self.theta_min else self.cfg["abs_floor_lph"]

        event_df = self.df[
            (self.df["date"] >= incident["start_day"].date())
            & (self.df["date"] <= incident["last_day"].date())
        ]
        avg_flow = event_df["flow"].mean()
        std_dev = event_df["flow"].std()

        # Dynamic thresholds
        fixture_thresh = 2 * baseline
        pipe_thresh = 5 * baseline
        burst_thresh = 10 * baseline

        if avg_flow <= fixture_thresh and std_dev < 0.2 * fixture_thresh:
            return (
                "Fixture Leak",
                f"Low, steady flow <{fixture_thresh:.0f} L/h. Likely toilets/taps.",
            )
        elif avg_flow <= pipe_thresh and std_dev < 0.3 * pipe_thresh:
            return (
                "Underground/Pipework Leak",
                f"Persistent steady flow <{pipe_thresh:.0f} L/h.",
            )
        elif avg_flow <= burst_thresh and std_dev >= 0.3 * pipe_thresh:
            return (
                "Appliance/Cycling Fault",
                f"Erratic pattern <{burst_thresh:.0f} L/h. Possible appliances.",
            )
        else:
            return (
                "Large Burst/Event",
                f"Very high flow >{burst_thresh:.0f} L/h. Likely major pipe break.",
            )

    def plot_leak_event(self, incident, site_cfg):
        """
        Enhanced leak event plot:
        - Shows pre-leak midnight flow (blue)
        - Shows confirmation period (orange)
        - Marks the day leak is confirmed
        """
        start_time = pd.Timestamp(incident["start_day"])
        end_time = pd.Timestamp(incident["last_day"])
        save_dir = self.cfg["save_dir"]

        # Define plot window: ±10 days
        plot_start = start_time - timedelta(days=10)
        plot_end = end_time + timedelta(days=10)
        plot_data = self.df[
            (self.df.index >= plot_start) & (self.df.index <= plot_end)
        ].copy()
        if plot_data.empty:
            logging.warning(
                f"No data to plot for {self.site_id} on {start_time.date()}"
            )
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Flow trace
        ax.plot(
            plot_data.index,
            plot_data["flow"],
            label="Hourly Flow (L/h)",
            color="dodgerblue",
            alpha=0.7,
        )

        # MNF threshold
        ax.axhline(
            y=site_cfg["theta_min"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"MNF Threshold ({site_cfg['theta_min']:.1f} L/h)",
        )

        # Pre-leak midnight flow (blue shade, 2 days before start)
        for offset in range(2, 0, -1):
            pre_day = (start_time - timedelta(days=offset)).floor("D")
            t0 = pre_day.replace(hour=0, minute=0)
            t1 = t0 + timedelta(hours=4)
            ax.axvspan(
                t0,
                t1,
                color="skyblue",
                alpha=0.3,
                label="Pre-leak midnight flow" if offset == 2 else None,
            )

        # Leak window (red shade)
        ax.axvspan(
            start_time, end_time, color="red", alpha=0.2, label="Red = Confirmed Leak"
        )

        # Confirmation period (orange shade)
        days_needed = max(
            3,
            self.get_persistence_needed(
                incident["max_deltaNF"],
                len(incident["reason_codes"]),
                incident["confidence"],
            ),
        )
        computed_alert = start_time + timedelta(days=days_needed - 1)
        alert_date = pd.to_datetime(incident.get("alert_date", computed_alert))

        ax.axvspan(
            start_time,
            alert_date,
            color="orange",
            alpha=0.3,
            label=f"Confirmation period ({days_needed} days)",
        )

        ax.axvline(
            alert_date,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Leak confirmed (alert date)",
        )

        # Labels & formatting
        ax.set_title(
            f"Site: {self.site_id} - Leak Event\n{start_time.date()} to {end_time.date()}"
        )
        ax.set_ylabel("Flow (L/h)")
        ax.set_xlabel("Timestamp")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        # Save
        folder = os.path.join(save_dir, self.site_id)
        os.makedirs(folder, exist_ok=True)
        filename = f"Leak_{self.site_id}_{start_time.strftime('%Y-%m-%d')}.png"
        plt.savefig(os.path.join(folder, filename))
        plt.close()
        logging.info(f"Enhanced plot saved: {os.path.join(folder, filename)}")

    def state_machine(self):

        if (
            self.daily is None
            or not isinstance(self.daily, pd.DataFrame)
            or self.daily.empty
        ):
            self.baselining()
        days = sorted(self.daily.index)
        if not days:
            return {}

        # Configs
        site_cfg = self.get_adaptive_threshold()
        merge_gap_days = int(self.cfg.get("merge_gap_days", 2))

        # Helpers
        def sev_rank(s):
            try:
                return int(str(s).lstrip("S"))
            except Exception:
                return 1

        daily_outputs = {}
        active = None
        last_committed = None  # last item in self.incidents

        for i, d in enumerate(days):
            # Wait until we have enough baseline history
            if (d - days[0]).days < self.cfg["baseline_window_days"]:
                daily_outputs[d] = {"status": "OK", "next_action": "None"}
                continue

            # ✅ FIX: Check if we already have signal components for this date
            # If yes, use the stored values to prevent recalculation from changed baselines
            d_key = d.strftime("%Y-%m-%d")
            if d_key in self.signal_components_by_date:
                # Use previously calculated signals
                cached = self.signal_components_by_date[d_key]
                sub_scores = cached["sub_scores"]
                leak_score = cached["leak_score"]
                deltaNF = cached["deltaNF"]
                NF_MAD = cached["NF_MAD"]
                logging.debug(
                    f"[{self.site_id}] {d_key}: Using CACHED signals - deltaNF={deltaNF:.1f}"
                )
            else:
                # Calculate fresh and store
                sub_scores, leak_score, deltaNF, NF_MAD = self.signals_and_score(d)
                self.signal_components_by_date[d_key] = {
                    "sub_scores": sub_scores.copy(),
                    "leak_score": leak_score,
                    "deltaNF": deltaNF,
                    "NF_MAD": NF_MAD,
                }
                logging.debug(
                    f"[{self.site_id}] {d_key}: CALCULATED FRESH signals - deltaNF={deltaNF:.1f}"
                )

            severity = self.get_severity(deltaNF)
            signals_fired = [k for k, v in sub_scores.items() if v > 0]

            # Pattern suppression (episodic fills)
            suppress = False
            if "BURSTBF" in signals_fired and sub_scores["BURSTBF"] > 0:
                nd = d + timedelta(days=1)
                if nd in self.daily.index:
                    nd_base, nd_mad = self.get_rolling_baseline(
                        nd, "NF_d", self.daily["NF_d"]
                    )
                    nd_delta = self.daily.loc[nd, "NF_d"] - nd_base
                    if nd_delta <= site_cfg["theta_min"]:
                        suppress = True
                        self.pattern_suppressions[d] = "EPISODIC FILL"

            if suppress:
                daily_outputs[d] = {
                    "status": "OK",
                    "next_action": "None",
                    "suppressed": self.pattern_suppressions[d],
                }
                continue

            # Confidence & persistence projection
            persistence_days = 1 if active is None else active["days_persisted"] + 1

            # ✅ FIX: Use FROZEN confidence from detector cache if available
            # The detector cache is pre-populated with frozen values from previous runs
            if d_key in self.confidence_by_date:
                confidence = self.confidence_by_date[d_key]
                logging.info(
                    f"[{self.site_id}] {d_key}: Using FROZEN confidence={confidence:.1f}%"
                )
            else:
                # Calculate fresh and freeze it
                confidence = self.get_confidence(
                    sub_scores, persistence_days, deltaNF, NF_MAD
                )
                self.confidence_by_date[d_key] = confidence
                logging.info(
                    f"[{self.site_id}] {d_key}: CALCULATED NEW confidence={confidence:.1f}% (now frozen)"
                )

            # Trigger condition
            trigger = (leak_score >= 30) or (
                sev_rank(severity) > 1 and deltaNF > site_cfg["theta_min"]
            )

            if trigger:
                if active:

                    # contiguous day → extend
                    if (d - active["last_day"]).days == 1:
                        active["last_day"] = d
                        active["days_persisted"] += 1

                    # small gap → merge into active
                    elif 1 < (d - active["last_day"]).days <= merge_gap_days:
                        gap = (
                            d - active["last_day"]
                        ).days - 1  # days with no data but within merge window
                        active["days_persisted"] += 1 + max(0, gap)
                        active["last_day"] = d

                    else:
                        # Try merging with the last committed incident if within gap
                        if (
                            last_committed
                            and 0
                            < (d - last_committed["last_day"]).days
                            <= merge_gap_days
                        ):
                            gap = (d - last_committed["last_day"]).days - 1
                            active = last_committed
                            active["days_persisted"] += 1 + max(0, gap)
                            active["last_day"] = d
                        else:
                            # start fresh
                            active = {
                                "site_id": self.site_id,
                                "status": "WATCH",
                                "start_day": d,
                                "last_day": d,
                                "max_deltaNF": 0.0,
                                "severity_max": "S1",
                                "days_persisted": 1,
                                "reason_codes": set(),
                                "volume_lost_kL": 0.0,
                                "confidence": 0.0,
                                # ✅ FIX: Store signal components by date for consistent confidence recalculation
                                "signal_components_by_date": {},
                            }
                            self.incidents.append(active)
                            last_committed = active

                    # Update attributes
                    active["max_deltaNF"] = max(active["max_deltaNF"], float(deltaNF))
                    if sev_rank(severity) > sev_rank(active["severity_max"]):
                        active["severity_max"] = severity
                    active["reason_codes"] |= set(signals_fired)
                    active["volume_lost_kL"] += float(deltaNF) * 24 / 1000.0

                    # ✅ FIX: Store signal components for this date in the incident
                    # This allows us to recalculate confidence with correct persistence later
                    if "signal_components_by_date" not in active:
                        active["signal_components_by_date"] = {}
                    active["signal_components_by_date"][d_key] = {
                        "sub_scores": sub_scores.copy(),
                        "deltaNF": float(deltaNF),
                        "NF_MAD": float(NF_MAD),
                        "confidence": float(
                            confidence
                        ),  # Store frozen confidence value
                    }

                    # Update main confidence to always be the LATEST day's value
                    active["confidence"] = float(confidence)

                    logging.info(
                        f"[{self.site_id}] Incident {active['start_day'].date()} -> {d_key}: "
                        f"persist={persistence_days}, conf={confidence:.1f}%, deltaNF={deltaNF:.1f}, "
                        f"NF_MAD={NF_MAD:.1f}, sub_scores={sub_scores}, "
                        f"signal_comp_count={len(active['signal_components_by_date'])}"
                    )

                    # (Re)compute persistence gate and alert_date EVERY day
                    needed = self.get_persistence_needed(
                        active["max_deltaNF"],
                        len(active["reason_codes"]),
                        active["confidence"],
                    )
                    active["alert_date"] = pd.to_datetime(
                        active["start_day"]
                    ) + timedelta(days=needed - 1)

                    # Escalation
                    if active["days_persisted"] >= needed:
                        if sev_rank(active["severity_max"]) <= 3:
                            active["status"] = "INVESTIGATE"
                        if sev_rank(active["severity_max"]) >= 4 or (
                            sev_rank(active["severity_max"]) >= 2
                            and active["confidence"] >= 70
                        ):
                            active["status"] = "CALL"

                else:
                    # Start first incident
                    active = {
                        "site_id": self.site_id,
                        "status": "WATCH",
                        "start_day": d,
                        "last_day": d,
                        "max_deltaNF": float(deltaNF),
                        "severity_max": severity,
                        "days_persisted": 1,
                        "reason_codes": set(signals_fired),
                        "volume_lost_kL": float(deltaNF) * 24 / 1000.0,
                        "confidence": float(confidence),
                        # ✅ FIX: Store signal components by date for consistent confidence recalculation
                        "signal_components_by_date": {
                            d_key: {
                                "sub_scores": sub_scores.copy(),
                                "deltaNF": float(deltaNF),
                                "NF_MAD": float(NF_MAD),
                                "confidence": float(
                                    confidence
                                ),  # Store frozen confidence value
                            }
                        },
                    }
                    needed = self.get_persistence_needed(
                        deltaNF, len(signals_fired), confidence
                    )
                    active["alert_date"] = pd.to_datetime(d) + timedelta(
                        days=needed - 1
                    )
                    self.incidents.append(active)
                    last_committed = active

                # Closure (night-flow based) — but skip for confirmed incidents
                if active and (active["status"] not in ("INVESTIGATE", "CALL")):
                    close_thresh = max(3 * NF_MAD, site_cfg["theta_min"])
                    if deltaNF <= close_thresh:
                        active["close_reason"] = "self-resolved/benign"
                        active = None

            else:
                # No trigger today ⇒ close only non-confirmed actives
                if active and (active["status"] not in ("INVESTIGATE", "CALL")):
                    active["close_reason"] = "self-resolved/benign"
                    active = None

            # Daily UI record
            status = active["status"] if active else "OK"
            daily_outputs[d] = {
                "status": status,
                "severity": (active["severity_max"] if active else None),
                "confidence": float(confidence),
                "deltaNF": float(deltaNF),
                "days_persisted": (active["days_persisted"] if active else 0),
                "est_volume_lost_kL": (active["volume_lost_kL"] if active else 0.0),
                "reason_codes": (
                    ", ".join(sorted(active["reason_codes"])) if active else ""
                ),
                "next_action": (
                    "Monitor next night"
                    if status == "WATCH"
                    else (
                        "Caretaker walk-through"
                        if status == "INVESTIGATE"
                        else "Escalate to plumber" if status == "CALL" else "None"
                    )
                ),
            }

        # Persist for reuse (avoid recompute)
        self.daily_outputs = daily_outputs

        # ✅ DEBUG: Log final incident state
        for inc in self.incidents:
            inc_id = inc.get(
                "event_id", f"{inc.get('start_day')}_{inc.get('last_day')}"
            )
            has_sig = "signal_components_by_date" in inc
            sig_count = len(inc.get("signal_components_by_date", {}))
            logging.info(
                f"[STATE_MACHINE_END] {self.site_id} - {inc_id}: "
                f"has_signal_components={has_sig}, count={sig_count}"
            )

        return daily_outputs

    def to_plotly_figs(self, incident, window_days=30):
        """
        Generate enhanced professional charts for national dashboard use.

        Returns 4 figures:
        1. Anomaly Timeline (replaces simple flow chart)
        2. MNF Control Chart (replaces simple night flow trend)
        3. After-Hours Breakdown (replaces simple after-hours trend)
        4. Weekly Heatmap (enhanced version)
        """
        start = pd.to_datetime(incident["start_day"])
        end = pd.to_datetime(incident["last_day"])
        alert_date = pd.to_datetime(incident.get("alert_date", end))

        half_window = window_days // 2
        window_start = start - timedelta(days=half_window)
        window_end = end + timedelta(days=half_window)

        # Filter daily summaries
        daily_window = self.daily[
            (self.daily.index >= window_start) & (self.daily.index <= window_end)
        ]

        # Filter raw hourly data
        df_window = self.df[
            (self.df.index >= window_start) & (self.df.index <= window_end)
        ]

        # 1. Enhanced Anomaly Timeline
        flow_fig = self._create_anomaly_timeline(
            df_window, daily_window, incident, start, end, alert_date
        )

        # 2. MNF Control Chart
        mnf_fig = self._create_mnf_control_chart(
            daily_window, incident, start, end, alert_date
        )

        # 3. After-Hours Breakdown
        ah_fig = self._create_after_hours_breakdown(
            df_window, daily_window, incident, start, end, alert_date
        )

        # 4. Enhanced Weekly Heatmap
        heatmap = self._create_enhanced_heatmap(df_window, start, end)

        return flow_fig, mnf_fig, ah_fig, heatmap

    def _create_anomaly_timeline(
        self, df_window, daily_window, incident, start, end, alert_date
    ):
        """
        Simplified flow timeline for non-technical users.
        Shows water usage over time with clear indication of the leak period.
        """

        # Calculate baseline (normal) flow level
        baseline_nf = self.theta_min if self.theta_min else self.cfg["abs_floor_lph"]

        # Get the maximum flow during the leak for scaling
        leak_mask = (df_window.index >= start) & (df_window.index <= end)
        max_leak_flow = (
            df_window.loc[leak_mask, "flow"].max()
            if leak_mask.any()
            else baseline_nf * 2
        )

        fig = go.Figure()

        # Split data into before leak, during leak, and after leak for clearer visualization
        before_leak = df_window[df_window.index < start]
        during_leak = df_window[(df_window.index >= start) & (df_window.index <= end)]
        after_leak = df_window[df_window.index > end]

        # 1. Normal period flow (before leak) - Blue
        if not before_leak.empty:
            fig.add_trace(
                go.Scatter(
                    x=before_leak.index,
                    y=before_leak["flow"],
                    mode="lines",
                    name="Normal Usage",
                    line=dict(color="rgb(100,180,255)", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(100,180,255,0.2)",
                    hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Water Flow: %{y:.0f} L/h<extra></extra>",
                )
            )

        # 2. Leak period flow - Red (highlighted)
        if not during_leak.empty:
            fig.add_trace(
                go.Scatter(
                    x=during_leak.index,
                    y=during_leak["flow"],
                    mode="lines",
                    name="🚨 Leak Period",
                    line=dict(color="rgb(255,80,80)", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(255,80,80,0.3)",
                    hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>⚠️ Water Flow: %{y:.0f} L/h<extra></extra>",
                )
            )

        # 3. After leak period (if any) - Blue
        if not after_leak.empty:
            fig.add_trace(
                go.Scatter(
                    x=after_leak.index,
                    y=after_leak["flow"],
                    mode="lines",
                    name="After Leak",
                    line=dict(color="rgb(100,180,255)", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(100,180,255,0.2)",
                    hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Water Flow: %{y:.0f} L/h<extra></extra>",
                    showlegend=False,
                )
            )

        # Add a simple "Normal Level" reference line
        fig.add_hline(
            y=baseline_nf,
            line_dash="dash",
            line_color="rgba(0,200,100,0.8)",
            line_width=2,
            annotation_text=f"✓ Normal Level ({baseline_nf:.0f} L/h)",
            annotation_position="top left",
            annotation=dict(
                font_size=12,
                bgcolor="rgba(0,100,50,0.8)",
                font_color="white",
                yshift=15,
            ),
        )

        # Add leak start marker with annotation
        fig.add_vline(
            x=start, line_color="rgba(255,165,0,0.8)", line_width=2, line_dash="dot"
        )
        fig.add_annotation(
            x=start,
            y=max_leak_flow * 0.95,
            text="⚠️ Leak Started",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange",
            font=dict(size=11, color="orange"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="orange",
            borderwidth=1,
        )

        # Add alert confirmation marker
        fig.add_vline(
            x=alert_date,
            line_color="rgba(255,50,50,0.9)",
            line_width=3,
            line_dash="solid",
        )
        fig.add_annotation(
            x=alert_date,
            y=max_leak_flow * 0.8,
            text="🔔 Alert Raised",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(size=11, color="white"),
            bgcolor="rgba(200,0,0,0.8)",
            bordercolor="red",
            borderwidth=1,
        )

        # Calculate water loss for display
        delta_nf = incident.get("max_deltaNF", 0)
        vol_lost = incident.get("volume_lost_kL", incident.get("ui_total_volume_kL", 0))

        # Simple, clear layout
        fig.update_layout(
            title=dict(
                text=f"<b>💧 Water Usage Timeline</b><br>"
                f"<span style='font-size:12px;color:#aaa'>"
                f"Leak detected: {start.strftime('%b %d')} → {end.strftime('%b %d, %Y')} | "
                f"Extra flow: ~{delta_nf:.0f} L/h | Water lost: ~{vol_lost:.1f} kL</span>",
                font=dict(size=16, color="white"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                tickformat="%b %d",
            ),
            yaxis=dict(
                title="Water Flow (Litres per Hour)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                rangemode="tozero",
            ),
            template="plotly_dark",
            hovermode="x unified",
            autosize=True,
            margin=dict(l=70, r=30, t=100, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(30,30,30,0.9)",
                font=dict(size=11),
            ),
            paper_bgcolor="rgba(30,30,30,1)",
            plot_bgcolor="rgba(25,25,35,1)",
        )

        return fig

    def _create_mnf_control_chart(self, daily_window, incident, start, end, alert_date):
        """
        Simplified Night Flow comparison chart for non-technical users.
        Shows daily night-time water usage with clear normal vs abnormal indication.
        """

        # Calculate baseline (normal) from pre-incident data
        pre_incident = self.daily[self.daily.index < start]
        if len(pre_incident) < 10:
            pre_incident = self.daily[self.daily.index < end]

        baseline = (
            self.robust_median(pre_incident["NF_d"]) if len(pre_incident) > 0 else 0
        )

        # Define simple threshold: 50% above baseline is concerning
        concern_level = baseline * 1.5

        fig = go.Figure()

        # Split data: before leak, during leak
        before_leak = daily_window[daily_window.index < start]
        during_leak = daily_window[
            (daily_window.index >= start) & (daily_window.index <= end)
        ]
        after_leak = daily_window[daily_window.index > end]

        # Normal period bars (before leak) - Blue
        if not before_leak.empty:
            fig.add_trace(
                go.Bar(
                    x=before_leak.index,
                    y=before_leak["NF_d"],
                    name="Normal Period",
                    marker_color="rgb(100,180,255)",
                    hovertemplate="<b>%{x|%b %d}</b><br>Night Flow: %{y:.0f} L/h<extra></extra>",
                )
            )

        # Leak period bars - Red
        if not during_leak.empty:
            fig.add_trace(
                go.Bar(
                    x=during_leak.index,
                    y=during_leak["NF_d"],
                    name="🚨 Leak Period",
                    marker_color="rgb(255,80,80)",
                    hovertemplate="<b>%{x|%b %d}</b><br>⚠️ Night Flow: %{y:.0f} L/h<extra></extra>",
                )
            )

        # After leak bars (if any) - Blue
        if not after_leak.empty:
            fig.add_trace(
                go.Bar(
                    x=after_leak.index,
                    y=after_leak["NF_d"],
                    name="After Leak",
                    marker_color="rgb(100,180,255)",
                    showlegend=False,
                    hovertemplate="<b>%{x|%b %d}</b><br>Night Flow: %{y:.0f} L/h<extra></extra>",
                )
            )

        # Normal baseline reference
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="rgba(0,200,100,0.8)",
            line_width=2,
            annotation_text=f"✓ Normal ({baseline:.0f} L/h)",
            annotation_position="top left",
            annotation=dict(
                font_size=12,
                bgcolor="rgba(0,100,50,0.8)",
                font_color="white",
                yshift=15,
            ),
        )

        # Calculate average during leak
        if not during_leak.empty:
            leak_avg = during_leak["NF_d"].mean()
            increase = leak_avg - baseline
            increase_pct = (increase / baseline * 100) if baseline > 0 else 0

            # Add annotation showing the increase
            fig.add_annotation(
                x=during_leak.index[len(during_leak) // 2],
                y=leak_avg,
                text=f"📈 +{increase:.0f} L/h<br>(+{increase_pct:.0f}% above normal)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(size=11, color="white"),
                bgcolor="rgba(200,0,0,0.85)",
                bordercolor="red",
                borderwidth=1,
            )

        fig.update_layout(
            title=dict(
                text="<b>🌙 Night-Time Water Usage (12am-4am)</b>",
                font=dict(size=14, color="white"),
                x=0.5,
                xanchor="center",
                y=0.95,
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                tickformat="%b %d",
            ),
            yaxis=dict(
                title="Night Flow (L/h)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                rangemode="tozero",
            ),
            template="plotly_dark",
            autosize=True,
            hovermode="x unified",
            margin=dict(l=60, r=30, t=70, b=50),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(30,30,30,0.9)",
            ),
            bargap=0.15,
            paper_bgcolor="rgba(30,30,30,1)",
            plot_bgcolor="rgba(25,25,35,1)",
        )

        return fig

    def _create_after_hours_breakdown(
        self, df_window, daily_window, incident, start, end, alert_date
    ):
        """
        Simplified After-Hours usage chart for non-technical users.
        Shows daily after-hours consumption with clear comparison to expected levels.
        """

        # Aggregate daily after-hours usage (simplified: just total after hours)
        daily_data = []
        window_start = df_window.index.min()
        window_end = df_window.index.max()

        for d in pd.date_range(window_start.date(), window_end.date(), freq="D"):
            day_data = df_window[df_window.index.date == d.date()]

            if len(day_data) == 0:
                continue

            # After hours: before 7am OR after 4pm
            after_hours = (
                day_data[(day_data.index.hour < 7) | (day_data.index.hour >= 16)][
                    "flow"
                ].sum()
                / 1000
            )  # Convert to kL

            is_leak = start.date() <= d.date() <= end.date()

            daily_data.append(
                {
                    "date": d,
                    "after_hours_kL": after_hours,
                    "is_leak": is_leak,
                }
            )

        if not daily_data:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template="plotly_dark", autosize=True)
            return fig

        df_daily = pd.DataFrame(daily_data)

        # Calculate baseline from non-leak days
        normal_days = df_daily[~df_daily["is_leak"]]
        baseline_kL = (
            normal_days["after_hours_kL"].median() if len(normal_days) > 0 else 0
        )

        fig = go.Figure()

        # Normal period bars - Blue
        normal_data = df_daily[~df_daily["is_leak"]]
        if not normal_data.empty:
            fig.add_trace(
                go.Bar(
                    x=normal_data["date"],
                    y=normal_data["after_hours_kL"],
                    name="Normal Days",
                    marker_color="rgb(100,180,255)",
                    hovertemplate="<b>%{x|%b %d}</b><br>After-Hours: %{y:.1f} kL<extra></extra>",
                )
            )

        # Leak period bars - Red
        leak_data = df_daily[df_daily["is_leak"]]
        if not leak_data.empty:
            fig.add_trace(
                go.Bar(
                    x=leak_data["date"],
                    y=leak_data["after_hours_kL"],
                    name="🚨 Leak Period",
                    marker_color="rgb(255,80,80)",
                    hovertemplate="<b>%{x|%b %d}</b><br>⚠️ After-Hours: %{y:.1f} kL<extra></extra>",
                )
            )

        # Normal baseline reference
        fig.add_hline(
            y=baseline_kL,
            line_dash="dash",
            line_color="rgba(0,200,100,0.8)",
            line_width=2,
            annotation_text=f"✓ Normal ({baseline_kL:.1f} kL/day)",
            annotation_position="top left",
            annotation=dict(
                font_size=12, bgcolor="rgba(0,100,50,0.8)", font_color="white"
            ),
        )

        # Calculate and show excess during leak
        if not leak_data.empty:
            total_leak = leak_data["after_hours_kL"].sum()
            expected = baseline_kL * len(leak_data)
            excess = total_leak - expected

            if excess > 0:
                # Add summary annotation
                fig.add_annotation(
                    x=leak_data["date"].iloc[-1],
                    y=leak_data["after_hours_kL"].max() * 1.1,
                    text=f"💧 Extra water used:<br><b>{excess:.1f} kL</b><br>(~${excess * 2:.0f} cost)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(size=11, color="white"),
                    bgcolor="rgba(200,0,0,0.85)",
                    bordercolor="red",
                    borderwidth=1,
                )

        fig.update_layout(
            title=dict(
                text="<b>🕐 After-Hours Water Usage (Before 7am & After 4pm)</b>",
                font=dict(size=14, color="white"),
                x=0.5,
                xanchor="center",
                y=0.95,
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                tickformat="%b %d",
            ),
            yaxis=dict(
                title="Water Used (kL)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                rangemode="tozero",
            ),
            template="plotly_dark",
            autosize=True,
            hovermode="x unified",
            margin=dict(l=60, r=30, t=70, b=70),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(30,30,30,0.9)",
            ),
            bargap=0.15,
            paper_bgcolor="rgba(30,30,30,1)",
            plot_bgcolor="rgba(25,25,35,1)",
        )

        return fig

    def _create_enhanced_heatmap(self, df_window, start, end):
        """
        Simplified Pattern Analysis for non-technical users.
        Shows when water is being used unexpectedly with clear visual indicators.
        """

        df_reset = df_window.reset_index()
        df_reset["time"] = pd.to_datetime(df_reset["time"])

        # Split into normal period and leak period
        normal_data = df_reset[df_reset["time"] < start]
        leak_data = df_reset[(df_reset["time"] >= start) & (df_reset["time"] <= end)]

        # Create time period categories (simplified from 24 hours)
        def get_time_period(hour):
            if 0 <= hour < 4:
                return "🌙 Night (12am-4am)"
            elif 4 <= hour < 7:
                return "🌅 Early Morning (4am-7am)"
            elif 7 <= hour < 16:
                return "☀️ Business Hours (7am-4pm)"
            else:
                return "🌆 Evening (4pm-12am)"

        time_periods = [
            "🌙 Night (12am-4am)",
            "🌅 Early Morning (4am-7am)",
            "☀️ Business Hours (7am-4pm)",
            "🌆 Evening (4pm-12am)",
        ]

        # Calculate average flow by time period for normal vs leak
        def calc_period_averages(data):
            if data.empty:
                return {p: 0 for p in time_periods}
            data = data.copy()
            data["period"] = data["time"].dt.hour.apply(get_time_period)
            return data.groupby("period")["flow"].mean().to_dict()

        normal_avgs = calc_period_averages(normal_data)
        leak_avgs = calc_period_averages(leak_data)

        # Prepare data for grouped bar chart
        normal_values = [normal_avgs.get(p, 0) for p in time_periods]
        leak_values = [leak_avgs.get(p, 0) for p in time_periods]

        fig = go.Figure()

        # Normal period bars
        fig.add_trace(
            go.Bar(
                name="Normal Period",
                x=time_periods,
                y=normal_values,
                marker_color="rgb(100,180,255)",
                text=[f"{v:.0f}" for v in normal_values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Normal: %{y:.0f} L/h<extra></extra>",
            )
        )

        # Leak period bars
        fig.add_trace(
            go.Bar(
                name="🚨 Leak Period",
                x=time_periods,
                y=leak_values,
                marker_color="rgb(255,80,80)",
                text=[f"{v:.0f}" for v in leak_values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Leak Period: %{y:.0f} L/h<extra></extra>",
            )
        )

        # Calculate the biggest increase
        increases = []
        for i, period in enumerate(time_periods):
            normal_val = normal_values[i]
            leak_val = leak_values[i]
            if normal_val > 0:
                pct_increase = ((leak_val - normal_val) / normal_val) * 100
            else:
                pct_increase = 100 if leak_val > 0 else 0
            increases.append((period, leak_val - normal_val, pct_increase))

        # Find the period with biggest absolute increase
        max_increase = max(increases, key=lambda x: x[1])

        # Add annotation for biggest problem area
        if max_increase[1] > 0:
            problem_idx = time_periods.index(max_increase[0])
            fig.add_annotation(
                x=max_increase[0],
                y=leak_values[problem_idx] * 1.15,
                text=f"⚠️ Biggest increase!<br>+{max_increase[1]:.0f} L/h ({max_increase[2]:.0f}%)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(size=11, color="white"),
                bgcolor="rgba(200,0,0,0.85)",
                bordercolor="red",
                borderwidth=1,
            )

        # Add "expected to be low" annotation for night hours
        fig.add_annotation(
            x="🌙 Night (12am-4am)",
            y=(
                max(normal_values[0], leak_values[0]) * 0.5
                if max(normal_values[0], leak_values[0]) > 0
                else 50
            ),
            text="Should be<br>near zero",
            showarrow=False,
            font=dict(size=9, color="rgba(200,200,200,0.7)"),
            bgcolor="rgba(0,0,0,0.5)",
        )

        # Calculate summary stats
        total_normal = (
            sum(normal_values) * len(normal_data) / max(len(normal_data), 1)
            if normal_data.any
            else 0
        )
        total_leak = (
            sum(leak_values) * len(leak_data) / max(len(leak_data), 1)
            if not leak_data.empty
            else 0
        )

        fig.update_layout(
            title=dict(
                text="<b>📊 Water Usage Pattern: Normal vs Leak Period</b>",
                font=dict(size=14, color="white"),
                x=0.5,
                xanchor="center",
                y=0.97,
            ),
            xaxis=dict(
                title="",
                showgrid=False,
                tickangle=0,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="Avg Flow (L/h)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
                rangemode="tozero",
            ),
            barmode="group",
            template="plotly_dark",
            autosize=True,
            margin=dict(l=60, r=30, t=60, b=160),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(30,30,30,0.9)",
            ),
            bargap=0.2,
            bargroupgap=0.1,
            paper_bgcolor="rgba(30,30,30,1)",
            plot_bgcolor="rgba(25,25,35,1)",
        )

        # Add a text box with key insight
        night_increase = leak_values[0] - normal_values[0]
        if night_increase > 10:
            insight_text = f"💡 Key Finding: Night-time usage increased by {night_increase:.0f} L/h. This suggests a leak (buildings should have minimal water use at night)."
        elif max_increase[1] > 10:
            insight_text = f"💡 Key Finding: Biggest increase during {max_increase[0].split(' ')[1]} (+{max_increase[1]:.0f} L/h more than normal)."
        else:
            insight_text = (
                "💡 Usage patterns appear relatively normal across all time periods."
            )

        fig.add_annotation(
            x=0.5,
            y=-0.48,
            xref="paper",
            yref="paper",
            text=insight_text,
            showarrow=False,
            font=dict(size=10, color="rgba(200,200,200,0.9)"),
            bgcolor="rgba(50,50,70,0.8)",
            bordercolor="rgba(100,100,150,0.5)",
            borderwidth=1,
            borderpad=6,
            align="center",
        )

        return fig


def load_tafe_data(file_path: str) -> dict:
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        logging.error(f"Data file not found at: {file_path}")
        raise

    all_dfs = []
    for sheet in sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.columns = df.columns.str.strip()
            required_cols = [
                "Timestamp",
                "De-identified Property Name",
                "Sum of Usage (L)",
            ]
            if not set(required_cols).issubset(df.columns):
                continue
            df = df.dropna(subset=["Timestamp", "De-identified Property Name"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df = df.rename(
                columns={
                    "De-identified Property Name": "site_id",
                    "Timestamp": "time",
                    "Sum of Usage (L)": "flow",
                }
            )
            df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0)
            df["flow"] = df["flow"].clip(lower=0)
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Error processing sheet {sheet}: {e}")
            continue

    combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
        subset=["time", "site_id"]
    )
    school_dfs = {
        school: group[["time", "flow"]].sort_values("time").reset_index(drop=True)
        for school, group in combined_df.groupby("site_id")
    }
    logging.info(f"Successfully loaded and split data for {len(school_dfs)} sites.")
    return school_dfs


def check_data_frequency(df, site_id):
    time_diffs = df["time"].diff().dropna()
    median_freq = time_diffs.median()
    if median_freq > timedelta(hours=2):
        logging.error(
            f"{site_id}: Data frequency too irregular (median {median_freq}). Cannot proceed."
        )
        raise ValueError(f"Data frequency too irregular for {site_id}")
    elif median_freq > timedelta(hours=1.5):
        logging.warning(
            f"{site_id}: Data frequency irregular (median {median_freq}). Results may be unreliable."
        )
    else:
        logging.info(f"{site_id}: Data frequency OK (median {median_freq})")


def process_site(args):
    """
    Engine-side single-site runner used by the replay loop.

    Expects args like: (site_id, df_slice, cfg, [...optional...], up_to_date, prev_signal_components, prev_confidence_by_date)
    Returns: (site_id, detector, confirmed_df)

    Responsibilities:
      - Defensive normalization of df_slice (time dtype, sort, basic checks)
      - Instantiate SchoolLeakDetector and run preprocess/state_machine
      - Canonicalize incident schemas (start_time/end_time/event_id/alert_date)
      - Build a 'confirmed_df' table that downstream UI can consume
    """
    import logging
    import pandas as pd
    import numpy as np

    logger = globals().get("log", logging.getLogger("replay"))

    # -----------------------
    # Unpack & basic hygiene
    # -----------------------
    try:
        site_id, df_slice, cfg, *rest = args
    except Exception:
        logger.exception("process_site: invalid args shape: %r", args)
        return None, None, pd.DataFrame()

    # up_to_date is expected as the last positional item (if provided)
    # prev_signal_components and prev_confidence_by_date are the last two dict items
    up_to_date = None
    prev_signal_components = {}
    prev_confidence_by_date = {}

    if rest:
        # Look for dicts at the end (frozen confidence and signal components)
        for candidate in reversed(rest):
            if isinstance(candidate, dict) and any(
                k.startswith("202") for k in candidate.keys() if isinstance(k, str)
            ):
                # This looks like a date-keyed dict
                if not prev_confidence_by_date and all(
                    isinstance(v, (int, float)) for v in candidate.values()
                ):
                    prev_confidence_by_date = candidate
                elif not prev_signal_components:
                    prev_signal_components = candidate

        # Look for datetime (up_to_date)
        for candidate in reversed(rest):
            try:
                up_to_date = pd.to_datetime(candidate)
                break
            except Exception:
                continue

    # Defensive copy & column checks
    if df_slice is None or len(df_slice) == 0:
        logger.warning("%s: empty df_slice provided to process_site", site_id)
        return site_id, None, pd.DataFrame()

    df = df_slice.copy()

    # Ensure 'time' is present and is datetime
    if "time" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "time"})

    if "time" not in df.columns:
        logger.error("%s: df_slice missing 'time' column", site_id)
        return site_id, None, pd.DataFrame()

    # Safe datetime conversion (no chained assignment)
    df.loc[:, "time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    # Optional: quick data frequency check for logging observability
    try:
        med_step = df["time"].diff().median()
        logger.info("%s: Data frequency OK (median %s)", site_id, med_step)
    except Exception:
        logger.debug("%s: could not compute median step", site_id)

    # -------------
    # Detector run
    # -------------
    try:
        detector = SchoolLeakDetector(df, site_id, cfg, up_to_date=up_to_date)

        # ✅ FIX: Restore FROZEN confidence values BEFORE running state_machine
        # This prevents recalculation of historical dates
        if prev_confidence_by_date and hasattr(detector, "confidence_by_date"):
            detector.confidence_by_date = prev_confidence_by_date.copy()
            logger.info(
                "%s: Restored %d FROZEN confidence values BEFORE state_machine",
                site_id,
                len(prev_confidence_by_date),
            )

        if prev_signal_components and hasattr(detector, "signal_components_by_date"):
            detector.signal_components_by_date = prev_signal_components.copy()
            logger.info(
                "%s: Restored %d signal components BEFORE state_machine",
                site_id,
                len(prev_signal_components),
            )

        if hasattr(detector, "preprocess"):
            detector.preprocess()
        if hasattr(detector, "state_machine"):
            detector.state_machine()
    except Exception:
        logger.exception("%s: detector failed during preprocess/state_machine", site_id)
        return site_id, None, pd.DataFrame()

    # ------------------------------------------
    # Helpers to canonicalize incidents/tables
    # ------------------------------------------
    def _canonize_incident(inc: dict) -> dict:
        """Normalize keys & types for downstream UI."""
        inc = dict(inc) if not isinstance(inc, dict) else inc

        # Timestamps: accept legacy 'start_day'/'last_day'
        st = inc.get("start_time", inc.get("start_day"))
        et = inc.get("end_time", inc.get("last_day", st))

        try:
            st = pd.to_datetime(st) if st is not None else None
        except Exception:
            st = None
        try:
            et = pd.to_datetime(et) if et is not None else st
        except Exception:
            et = st

        inc["start_time"] = st
        inc["end_time"] = et

        # Alert date: if missing, derive from days_needed
        if inc.get("alert_date") is None and st is not None:
            dn = inc.get("days_needed")
            try:
                dn = int(dn) if dn is not None else None
            except Exception:
                dn = None
            if dn and dn > 0:
                # align with plotting/confirmation shading convention
                inc["alert_date"] = st.normalize() + pd.Timedelta(days=dn - 1)
            else:
                # fallback: use end_time or start_time
                inc["alert_date"] = et.normalize() if et is not None else st.normalize()

        # Stable event_id (site__YYYY-MM-DD__YYYY-MM-DD)
        if not inc.get("event_id") and st is not None:
            st_d = st.date()
            et_d = et.date() if et is not None else st_d
            inc["event_id"] = f"{site_id}__{st_d}__{et_d}"

        # Coerce some common numeric fields defensively
        for num_key in ("confidence", "max_deltaNF", "volume_lost_kL"):
            if num_key in inc and inc[num_key] is not None:
                try:
                    inc[num_key] = float(inc[num_key])
                except Exception:
                    pass

        # Ensure status exists (string) for filtering confirmed incidents
        if "status" not in inc or inc["status"] is None:
            # leave unset; downstream can treat as WATCH/UNKNOWN
            inc["status"] = inc.get("status", "UNKNOWN")

        return inc

    def _canonize_confirmed_df(df_in: pd.DataFrame) -> pd.DataFrame:
        """Add start_time/end_time/event_id columns if missing and ensure Timestamp types."""
        if df_in is None or len(df_in) == 0:
            return pd.DataFrame()

        out = df_in.copy()

        # start_time / end_time from legacy fields if needed
        if "start_time" not in out.columns and "start_day" in out.columns:
            out.loc[:, "start_time"] = pd.to_datetime(out["start_day"], errors="coerce")
        elif "start_time" in out.columns:
            out.loc[:, "start_time"] = pd.to_datetime(
                out["start_time"], errors="coerce"
            )

        if "end_time" not in out.columns and "last_day" in out.columns:
            out.loc[:, "end_time"] = pd.to_datetime(out["last_day"], errors="coerce")
        elif "end_time" in out.columns:
            out.loc[:, "end_time"] = pd.to_datetime(out["end_time"], errors="coerce")

        # alert_date if present
        if "alert_date" in out.columns:
            out.loc[:, "alert_date"] = pd.to_datetime(
                out["alert_date"], errors="coerce"
            )

        # event_id stable
        if "event_id" not in out.columns:
            st_ser = out.get("start_time")
            et_ser = out.get("end_time", st_ser)
            sid_ser = pd.Series([site_id] * len(out), index=out.index)

            def _mk_eid(row):
                st = row.get("start_time")
                et = (
                    row.get("end_time")
                    if pd.notna(row.get("end_time"))
                    else row.get("start_time")
                )
                try:
                    sd = pd.to_datetime(st).date()
                except Exception:
                    sd = None
                try:
                    ed = pd.to_datetime(et).date() if et is not None else sd
                except Exception:
                    ed = sd
                if sd is None:
                    return f"{site_id}__unknown__unknown"
                return f"{site_id}__{sd}__{ed}"

            out.loc[:, "event_id"] = [
                _mk_eid(out.loc[i, :].to_dict()) for i in out.index
            ]

        return out

    # -------------------------------------
    # Gather incidents & build confirmed_df
    # -------------------------------------
    incidents = []
    try:
        if hasattr(detector, "incidents") and detector.incidents:
            for inc in detector.incidents:
                try:
                    incidents.append(_canonize_incident(inc))
                except Exception:
                    logger.exception("%s: failed to canonize incident", site_id)
    except Exception:
        logger.exception("%s: accessing detector.incidents failed", site_id)

    # If detector provides a confirmed table, canonize it; otherwise derive a simple one
    confirmed_df = pd.DataFrame()
    try:
        if hasattr(detector, "confirmed_df") and detector.confirmed_df is not None:
            confirmed_df = _canonize_confirmed_df(detector.confirmed_df)
        else:
            # Derive lightweight confirmed table from incidents with status INVESTIGATE/CALL
            confirmed = [
                inc
                for inc in incidents
                if str(inc.get("status", "")).upper() in ("INVESTIGATE", "CALL")
            ]
            confirmed_df = pd.DataFrame(confirmed) if confirmed else pd.DataFrame()
            if not confirmed_df.empty:
                confirmed_df = _canonize_confirmed_df(confirmed_df)
    except Exception:
        logger.exception("%s: building confirmed_df failed", site_id)
        confirmed_df = pd.DataFrame()

    # Helpful trace when hitting a new confirmation "today"
    if up_to_date is not None and not confirmed_df.empty:
        try:
            today = pd.to_datetime(up_to_date).normalize()
            todays = confirmed_df.loc[
                confirmed_df.get("alert_date", pd.NaT).dt.normalize() == today
            ]
            for _, row in todays.iterrows():
                logger.info(
                    "Confirm @ %s | %s | sev=%s | conf=%s%%",
                    today.date(),
                    row.get("event_id"),
                    row.get("severity_max"),
                    (
                        f"{float(row.get('confidence', 0)):.0f}"
                        if pd.notna(row.get("confidence", np.nan))
                        else "?"
                    ),
                )
        except Exception:
            # non-fatal
            pass

    return site_id, detector, confirmed_df


def run_efficient_pipeline(
    school_dfs: dict, cfg: dict, leak_log_file=None, up_to_date=None
):
    total_schools = len(school_dfs)
    logging.info(
        f"--- Starting Efficient Leak Detection Pipeline for {total_schools} schools ---"
    )

    all_confirmed_leaks = []
    rejected_events = []

    # Load historical leak log if available
    leak_log = []
    if leak_log_file:
        try:
            leak_log = pd.read_csv(leak_log_file, parse_dates=["start", "end"]).to_dict(
                "records"
            )
            logging.info(
                f"Loaded {len(leak_log)} records from leak log: {leak_log_file}"
            )
        except FileNotFoundError:
            logging.warning(
                f"Leak log file not found at {leak_log_file}. Proceeding without leak log."
            )
        except Exception as e:
            logging.error(
                f"Error loading leak log file {leak_log_file}: {e}. Proceeding without leak log."
            )

    # Slice data if up_to_date
    if up_to_date:
        logging.info(f"Slicing data up to {pd.to_datetime(up_to_date).date()}")
        sliced_dfs = {
            sid: df[df["time"] <= pd.to_datetime(up_to_date)]
            for sid, df in school_dfs.items()
        }
    else:
        logging.info("Using full dataset (no cutoff applied).")
        sliced_dfs = school_dfs

    # Parallel processing across schools
    logging.info("Dispatching school datasets to worker processes...")
    with ProcessPoolExecutor() as executor:
        args_list = [
            (sid, df, cfg, leak_log, up_to_date) for sid, df in sliced_dfs.items()
        ]
        results = list(executor.map(process_site, args_list))

    site_detectors = {}
    site_confirmed_dfs = {}
    for idx, (school_id, detector, confirmed_df) in enumerate(results, start=1):
        if detector:
            logging.info(
                f"[{idx}/{total_schools}] Processed {school_id} "
                f"| {len(confirmed_df)} confirmed leaks detected"
            )
            site_detectors[school_id] = detector
            site_confirmed_dfs[school_id] = confirmed_df
            all_confirmed_leaks.extend(confirmed_df.to_dict("records"))
        else:
            logging.warning(
                f"[{idx}/{total_schools}] Skipped {school_id} (no detector returned)"
            )

    # Save confirmed leaks
    confirmed_df = pd.DataFrame(all_confirmed_leaks)
    export_path = os.path.join(cfg["export_folder"], "Efficient_Confirmed_Leaks.csv")
    os.makedirs(cfg["export_folder"], exist_ok=True)
    confirmed_df.to_csv(export_path, index=False)
    logging.info(
        f"✅ Completed pipeline: {len(confirmed_df)} confirmed leaks saved to {export_path}"
    )

    # Save rejected events if any
    if rejected_events:
        rejected_df = pd.DataFrame(rejected_events)
        rejected_path = os.path.join(cfg["export_folder"], "Rejected_Events.csv")
        rejected_df.to_csv(rejected_path, index=False)
        logging.info(
            f"⚠️ Rejected {len(rejected_events)} events. Saved to {rejected_path}"
        )

    # Save summary per site
    if not confirmed_df.empty:
        summary = (
            confirmed_df.groupby("site_id")
            .agg(
                num_leaks=("site_id", "size"),
                total_volume=("total_volume_L", "sum"),
                avg_duration=("duration_hours", "mean"),
            )
            .reset_index()
        )
        summary_path = os.path.join(cfg["export_folder"], "Leak_Summary.csv")
        summary.to_csv(summary_path, index=False)
        logging.info(f"📊 Summary report saved to {summary_path}")

    logging.info("--- Leak Detection Pipeline finished successfully ---")
    return confirmed_df


def validate_config(cfg):
    required_keys = [
        "night_start",
        "night_end",
        "after_hours_start",
        "after_hours_end",
        "baseline_window_days",
        "abs_floor_lph",
        "sustained_after_hours_delta_kl",
        "spike_multiplier",
        "spike_ref_percentile",
        "score_weights",
        "persistence_gates",
        "severity_bands_lph",
        "cusum_k",
        "cusum_h",
        "export_folder",
        "data_path",
        "save_dir",
    ]
    for key in required_keys:
        if key not in cfg:
            logging.error(f"Missing config key: {key}")
            raise KeyError(f"Missing config key: {key}")
    if cfg["night_start"] >= cfg["night_end"]:
        raise ValueError("night_start must be less than night_end")
    if cfg["abs_floor_lph"] <= 0:
        raise ValueError("abs_floor_lph must be positive")
    if not all(w > 0 for w in cfg["score_weights"].values()):
        raise ValueError("All score_weights must be positive")


# %%

if __name__ == "__main__":
    with open("config_leak_detection.yml", "r") as f:
        cfg = yaml.safe_load(f)
    logging.info("Configuration loaded from config_leak_detection.yml")
    validate_config(cfg)

    # Load all sites
    school_dfs = load_tafe_data(cfg["data_path"])

    # --- Pick one property only ---
    target_site = "Property 11127"
    single_school_dfs = {target_site: school_dfs[target_site]}

    # Run pipeline just for this property
    confirmed_leaks = run_efficient_pipeline(
        single_school_dfs, cfg, leak_log_file=cfg.get("leak_log_path")
    )

    print(f"\n--- Confirmed Leaks for {target_site} ---")
    if confirmed_leaks.empty:
        print("No confirmed leaks found.")
    else:
        print(confirmed_leaks.to_string())


# %%
