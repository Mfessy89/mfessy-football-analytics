# Football Match Forecast

This Streamlit app turns the supplied football prediction notebook into an
interactive forecast. It keeps the notebook's venue-aware expected-goals
calculation, Poisson outcome probabilities, likely scorelines, recent form, and
chronological backtest.

## Run

From the project root:

```streamlit run app.py
```

The app starts with `data/epl_2024_25.csv`, a bundled 2024–25 English Premier
League results file from [football-data.co.uk](https://www.football-data.co.uk/).
Use the upload control to select a newer compatible results CSV. It must include
`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, and `FTR` columns.

The sidebar also has a separate upcoming-fixtures uploader. Its CSV must include
`Date`, `HomeTeam`, and `AwayTeam`; it must not be used as a results file. The app
calculates each fixture using only completed results dated before that fixture.
Fixtures with duplicate rows, invalid dates, blank teams, unrecognized team names,
or insufficient earlier home/away history are reported and are not given invented
predictions. The batch forecast can be downloaded as a CSV.

The original notebook is kept at `Football_prediction.ipynb`.

## Model limitations

Expected goals are the average of each team's venue-specific goals scored and
conceded. Independent Poisson distributions produce the outcome probabilities.
The model does not include player availability, lineups, odds, or live events.
Predictions and backtest results are estimates, not guarantees.

## Tests

Run the model and CSV validation tests from the Python app directory:

```bash
uv run --project . python -m unittest discover -s python_app -p 'test_*.py'
```
