# Freight Rate Prediction

LightGBM lane model + small market adjustment. Forward-chaining (time-based) validation.

## Run
```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py          # CV, final fit, writes validation_predictions.csv + outputs/december_chart_inputs_completed.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs_completed.csv
python src/make_figures.py && node src/make_report.js   # optional: rebuild report (needs `npm i docx`)
```
Data files go in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Layout
- `src/freight_model.py` – features, data-quality handling, model
- `src/run_pipeline.py` – validation, final training, predictions
- `outputs/` – CV tables, completed December file, figures, report
- `scorer_results/candidate_december.png` – chart from the provided `score.py`

## Key decisions
- Split: train on earlier months, test on next month (Jun–Oct folds), because the target period (Nov–Dec) is in the future.
- `quote_signal` is excluded: it equals rate/distance in some months, is inverted in others, and is noise in Nov/Dec.
- Weight sign errors fixed (abs), missing kept as NaN, cap flagged; missing `market_index` filled by same-date median.
- Coordinates instead of city names so 8 unseen validation cities work.
- L1 loss on log(rate) for robustness to ~1.4% label spikes.
