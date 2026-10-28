"""End-to-end: forward-chaining CV -> final fit -> validation_predictions.csv + december chart inputs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from freight_model import FreightModel, lane_features, haversine_miles

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def metrics(y, p):
    return dict(MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)),
                MAPE=np.mean(np.abs(y - p) / y) * 100, MedAPE=np.median(np.abs(y - p) / y) * 100)


class NaiveRPM:        # baseline: distance x median rate-per-mile by equipment
    def fit(self, df):
        self.m = (df.posted_rate / df.distance).groupby(df.equipment).median(); return self
    def predict(self, df): return (df.distance * df.equipment.map(self.m)).values


def forward_cv(tr, make, folds=(6, 7, 8, 9, 10)):
    """Train on all months < k, test on month k (mimics predicting the *future* Nov-Dec)."""
    rows = []
    for k in folds:
        a, b = tr[tr.date.dt.month < k], tr[tr.date.dt.month == k]
        p = make().fit(a).predict(b); y = b.posted_rate.values
        r = {"test_month": k, **metrics(y, p)}
        ok = np.abs(np.log(y / p)) < 0.3          # same, excluding rows whose label is an extreme spike
        r.update({f"{k2}_noSpike": v for k2, v in metrics(y[ok], p[ok]).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    tmpl = pd.read_csv(DATA / "validation_predictions_template.csv")

    variants = {
        "Naive: distance x median $/mile by equipment": NaiveRPM,
        "LightGBM lane model, no date adjustment": lambda: FreightModel(use_date_adjust=False),
        "LightGBM + recent level (30d)": lambda: FreightModel(beta=0.0),
        "FINAL: LightGBM + recent level + market_index": lambda: FreightModel(),
        "Final + label-spike cleaning (|resid|>0.25)": lambda: FreightModel(clean_labels_thr=0.25),
    }
    summary, per_fold = [], {}
    for name, mk in variants.items():
        cv = forward_cv(tr, mk); per_fold[name] = cv
        summary.append({"model": name, **cv.drop(columns="test_month").mean().round(3).to_dict()})
        print(f"{name:48s} MAE={cv.MAE.mean():7.2f} RMSE={cv.RMSE.mean():7.1f} MAPE={cv.MAPE.mean():5.2f}% "
              f"| noSpike MAE={cv.MAE_noSpike.mean():6.2f} MAPE={cv.MAPE_noSpike.mean():5.2f}%")
    pd.DataFrame(summary).to_csv(OUT / "cv_summary.csv", index=False)
    per_fold["FINAL: LightGBM + recent level + market_index"].to_csv(OUT / "cv_final_per_fold.csv", index=False)

    # quote_signal ablation (documents why it was dropped): same LightGBM lane model with qs added
    import lightgbm as lgb
    from freight_model import LGB_PARAMS
    rows = []
    for k in (6, 7, 8, 9, 10):
        a, b = tr[tr.date.dt.month < k], tr[tr.date.dt.month == k]
        Xa, Xb = lane_features(a), lane_features(b)
        base = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xa, np.log(a.posted_rate))
        Xa2, Xb2 = Xa.assign(qs=a.quote_signal), Xb.assign(qs=b.quote_signal)
        withq = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xa2, np.log(a.posted_rate))
        rows.append({"test_month": k, "MAE_without_qs": metrics(b.posted_rate.values, np.exp(base.predict(Xb)))["MAE"],
                     "MAE_with_qs": metrics(b.posted_rate.values, np.exp(withq.predict(Xb2)))["MAE"]})
    pd.DataFrame(rows).to_csv(OUT / "cv_quote_signal_ablation.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    # ---------------- final model on ALL labelled data ----------------
    model = FreightModel().fit(tr)
    print(f"final: beta(market_index)={model.beta:.3f} level={model.level:+.4f} recent_mi={model.recent_mi:.3f}")
    pred = model.predict(va)
    out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va.load_id, "predicted_rate": np.round(pred, 2)}),
                                  on="load_id", how="left")
    assert out.predicted_rate.notna().all() and len(out) == 12_000 and (out.predicted_rate > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # ---------------- fixed December what-if ----------------
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])
    city = {}
    for side in ("pickup", "delivery"):
        for _, r in tr.drop_duplicates(side).iterrows():
            city[r[side]] = (r[f"{side}_lat"], r[f"{side}_lon"])
    p, d = city["Lexington"], city["Fort Wayne"]
    X = pd.DataFrame({"distance": dec.distance, "equipment": dec.equipment, "weight": dec.weight,
                      "pickup_lat": p[0], "pickup_lon": p[1], "delivery_lat": d[0], "delivery_lon": d[1],
                      "date": dec.date})
    # market_index for each December day = same-day median across the validation loads (features are given at inference)
    mi_val = va.groupby("date").market_index.median()
    dec["predicted_rate"] = np.round(model.predict(X, mi_by_date=mi_val), 2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(OUT / "december_chart_inputs_completed.csv", index=False)
    print(dec[["date", "predicted_rate"]].describe())

    # diagnostics for report
    json.dump({"beta": model.beta, "level": model.level, "recent_mi": model.recent_mi,
               "val_pred_mean": float(pred.mean()), "train_rate_mean": float(tr.posted_rate.mean())},
              open(OUT / "final_model_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
