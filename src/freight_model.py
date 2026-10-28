"""Freight-rate model.

Design (see REPORT for the reasoning):
  1. Lane model   - LightGBM, L1 loss on log(posted_rate). Uses distance, equipment, weight, geography
                    (coordinates, not city names, so unseen cities generalise). NO raw date features and
                    NO quote_signal (its relationship to the rate is regime dependent / leaks in some months).
  2. Date effect  - a small, explicit adjustment: recent price level (median residual of the last
                    `level_days` training days) + beta * (market_index - recent market_index).
                    beta is estimated from the daily residual series (regression on market_index + linear trend).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
WEIGHT_CAP = 47_500.0
LGB_PARAMS = dict(objective="regression_l1", n_estimators=700, learning_rate=0.04, num_leaves=31,
                  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1)


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def lane_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features. Data-quality handling is done here:
       - weight: sign errors fixed with abs(); missing weight stays NaN (LightGBM routes NaNs natively);
         readings at the +/-47,500 cap are flagged (the true weight is censored there).
    """
    X = pd.DataFrame(index=df.index)
    X["log_dist"] = np.log(df["distance"])
    X["equip"] = df["equipment"].map(EQUIP)
    w = df["weight"]
    X["weight"] = w.abs()
    X["w_neg"] = (w < 0).astype(int)
    X["w_capped"] = (w.abs() >= WEIGHT_CAP).astype(int)
    X["plat"], X["plon"] = df["pickup_lat"], df["pickup_lon"]
    X["dlat"], X["dlon"] = df["delivery_lat"], df["delivery_lon"]
    X["circuity"] = df["distance"] / haversine_miles(df["pickup_lat"], df["pickup_lon"],
                                                      df["delivery_lat"], df["delivery_lon"])
    return X


class FreightModel:
    def __init__(self, level_days: int = 30, beta: float | None = None, clean_labels_thr: float | None = None,
                 use_date_adjust: bool = True, params: dict | None = None):
        self.level_days, self.beta_fixed, self.clean_thr = level_days, beta, clean_labels_thr
        self.use_date_adjust = use_date_adjust
        self.params = {**LGB_PARAMS, **(params or {})}

    # ---- market index helpers -------------------------------------------------------------
    @staticmethod
    def _mi_filled(df, extra_mi_by_date=None):
        """market_index with the ~0.8% missing values filled from the same-date median (index is daily)."""
        by_date = df.groupby("date")["market_index"].median()
        if extra_mi_by_date is not None:
            by_date = by_date.combine_first(extra_mi_by_date)
        return df["market_index"].fillna(df["date"].map(by_date)), by_date

    def fit(self, df: pd.DataFrame) -> "FreightModel":
        y = np.log(df["posted_rate"].values)
        X = lane_features(df)
        keep = np.ones(len(df), bool)
        if self.clean_thr:                                   # optional: drop extreme label spikes from training
            g0 = lgb.LGBMRegressor(**self.params).fit(X, y)
            keep = np.abs(y - g0.predict(X)) < self.clean_thr
        self.gbm = lgb.LGBMRegressor(**self.params).fit(X[keep], y[keep])
        self.n_train_used = int(keep.sum())

        # --- date effect from daily median residuals (robust to spikes) ---
        mi, _ = self._mi_filled(df)
        daily = (pd.DataFrame({"date": df["date"].values, "res": y - self.gbm.predict(X), "mi": mi.values})
                 .groupby("date").median().reset_index().sort_values("date"))
        daily["t"] = (daily["date"] - daily["date"].min()).dt.days
        if self.beta_fixed is None:
            A = np.c_[np.ones(len(daily)), daily["mi"], daily["t"]]
            self.beta = float(np.linalg.lstsq(A, daily["res"].values, rcond=None)[0][1])
        else:
            self.beta = self.beta_fixed
        recent = daily.tail(self.level_days)
        self.recent_mi = float(recent["mi"].mean())
        self.level = float(recent["res"].mean())
        return self

    def predict(self, df: pd.DataFrame, mi_by_date: pd.Series | None = None) -> np.ndarray:
        """mi_by_date: optional date->market_index lookup for rows whose own market_index is missing / absent
        (used for the fixed December what-if inputs, where only the date is supplied)."""
        lane = self.gbm.predict(lane_features(df))
        if not self.use_date_adjust:
            return np.exp(lane)
        if "market_index" in df:
            mi, _ = self._mi_filled(df, mi_by_date)
        else:
            mi = df["date"].map(mi_by_date)
        mi = mi.fillna(self.recent_mi)                       # last-resort fallback = no market adjustment
        eff = self.level + self.beta * (mi.values - self.recent_mi)
        return np.exp(lane + eff)
