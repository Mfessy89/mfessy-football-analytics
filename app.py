from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from model import (
    MIN_TEAM_MATCHES,
    MatchDataError,
    OUTCOME_LABELS,
    predict_fixtures,
    predict_venue_goals,
    prepare_fixtures,
    prepare_matches,
    recent_team_matches,
    run_backtest,
    score_and_outcome_probabilities,
    team_history_counts,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "epl_2024_25.csv"
NOTEBOOK_PATH = APP_DIR / "Football_prediction.ipynb"
DATA_SOURCE_URL = "https://www.football-data.co.uk/englandm.php"

st.set_page_config(page_title="Football Match Forecast", page_icon=None, layout="wide")


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return prepare_matches(pd.read_csv(SAMPLE_DATA_PATH))


@st.cache_data(show_spinner=False)
def load_uploaded_data(content: bytes) -> tuple[pd.DataFrame, int, int]:
    frame = pd.read_csv(BytesIO(content))
    matches = prepare_matches(frame)
    ignored_rows = len(frame) - len(matches)
    duplicate_rows = int(
        matches.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()
    )
    return matches, ignored_rows, duplicate_rows


@st.cache_data(show_spinner=False)
def load_uploaded_fixtures(content: bytes) -> pd.DataFrame:
    return prepare_fixtures(pd.read_csv(BytesIO(content)))


@st.cache_data(show_spinner="Running a chronological backtest...")
def cached_backtest(matches: pd.DataFrame):
    return run_backtest(matches)


def render_recent_form(team: str, matches: pd.DataFrame, as_of: pd.Timestamp) -> None:
    form = recent_team_matches(team, matches, as_of)
    st.markdown(f"**{team}**")
    if form.empty:
        st.caption("No earlier results for this team.")
        return
    st.dataframe(form, hide_index=True, width="stretch")
    st.caption("Form, most recent last: " + " · ".join(form["Result"].tolist()))


st.title("Football Match Forecast")
st.caption(
    "Estimate expected goals and match outcomes from historical results using "
    "venue-aware averages and a Poisson model."
)

with st.sidebar:
    st.header("Match data")
    uploaded_csv = st.file_uploader(
        "Upload a football-data CSV",
        type=["csv"],
        help=(
            "Expected columns: Date, HomeTeam, AwayTeam, FTHG, FTAG and FTR. "
            "Uploading replaces the bundled example for this session."
        ),
    )

    if uploaded_csv is None:
        matches = load_sample_data()
        dataset_name = "Bundled Premier League 2024–25 results"
    else:
        try:
            matches, ignored_rows, duplicate_rows = load_uploaded_data(
                uploaded_csv.getvalue()
            )
            dataset_name = uploaded_csv.name
            if ignored_rows:
                st.warning(
                    f"Ignored {ignored_rows:,} historical row(s) with missing or "
                    "invalid dates, team names, or scores."
                )
            if duplicate_rows:
                st.warning(
                    f"Found {duplicate_rows:,} duplicate historical fixture row(s). "
                    "They remain in the dataset and will count toward team statistics."
                )
        except (
            MatchDataError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as error:
            st.error(f"Could not use that CSV: {error}")
            st.stop()

    fixture_upload = st.file_uploader(
        "Upload upcoming fixtures CSV",
        type=["csv"],
        key="upcoming_fixtures_csv",
        help="Required columns: Date, HomeTeam and AwayTeam. Do not include scores.",
    )
    fixtures = None
    fixture_upload_error = None
    if fixture_upload is not None:
        try:
            fixtures = load_uploaded_fixtures(fixture_upload.getvalue())
            st.caption(f"{len(fixtures):,} upcoming fixture(s) loaded.")
        except (
            MatchDataError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as error:
            fixture_upload_error = str(error)
            st.error(f"Could not use that fixtures CSV: {fixture_upload_error}")

    latest_data_date = matches["Date"].max()
    today_timestamp = pd.Timestamp(date.today())
    data_age_days = (today_timestamp.normalize() - latest_data_date.normalize()).days
    default_prediction_date = max(
        date.today(), (latest_data_date + pd.Timedelta(days=1)).date()
    )
    prediction_date = st.date_input(
        "Prediction date",
        value=default_prediction_date,
        help="Only completed matches before this date are used.",
    )
    as_of = pd.Timestamp(prediction_date)
    st.caption(
        f"{len(matches):,} completed matches · latest result "
        f"{latest_data_date:%d %b %Y}"
    )
    st.caption(
        "The bundled sample is historical, not a live fixture feed. Upload newer "
        "results to refresh the model's form."
    )
    if data_age_days > 180:
        st.warning(
            f"Historical results are {data_age_days} days old as of today; the "
            f"dataset ends on {latest_data_date:%d %b %Y}. Predictions may not "
            "reflect current team performance."
        )
    elif data_age_days < 0:
        st.warning(
            f"The results dataset contains dates after today (latest: "
            f"{latest_data_date:%d %b %Y}). Check that those rows are completed "
            "results, not unplayed fixtures."
        )
    elif (as_of - latest_data_date).days > 180:
        st.warning(
            f"The selected forecast date ({as_of:%d %b %Y}) is more than 180 days "
            "after the latest historical result."
        )

past_matches = matches[matches["Date"] < as_of].copy()
teams = sorted(set(past_matches["HomeTeam"]) | set(past_matches["AwayTeam"]))
eligible_teams = []
for team in teams:
    counts = team_history_counts(team, past_matches)
    if (
        counts["total_matches"] >= MIN_TEAM_MATCHES
        and counts["home_matches"] > 0
        and counts["away_matches"] > 0
    ):
        eligible_teams.append(team)

forecast_tab, fixtures_tab, evaluation_tab, data_tab = st.tabs(
    ["Match forecast", "Upcoming fixtures", "Model performance", "Data and notebook"]
)

with forecast_tab:
    st.subheader("Choose a fixture")
    st.write(f"Prediction date: **{prediction_date:%d %B %Y}**")

    if len(eligible_teams) < 2:
        st.warning(
            "There is not enough earlier data to forecast a fixture. Choose a later "
            "prediction date or upload a results file with more matches."
        )
    else:
        preferred_home = "Arsenal" if "Arsenal" in eligible_teams else eligible_teams[0]
        preferred_away = (
            "Chelsea"
            if "Chelsea" in eligible_teams and "Chelsea" != preferred_home
            else next(team for team in eligible_teams if team != preferred_home)
        )
        home_col, away_col = st.columns(2)
        with home_col:
            home_team = st.selectbox(
                "Home team",
                eligible_teams,
                index=eligible_teams.index(preferred_home),
            )
        away_options = [team for team in eligible_teams if team != home_team]
        with away_col:
            away_team = st.selectbox(
                "Away team",
                away_options,
                index=away_options.index(preferred_away)
                if preferred_away in away_options
                else 0,
            )

        estimates = predict_venue_goals(home_team, away_team, past_matches)
        if estimates is None:
            st.warning(
                "The selected teams need both home and away results before the "
                "prediction date."
            )
        else:
            if (
                estimates["home_matches"] < MIN_TEAM_MATCHES
                or estimates["away_matches"] < MIN_TEAM_MATCHES
            ):
                st.warning(
                    f"At least {MIN_TEAM_MATCHES} earlier matches per team are "
                    "recommended. Treat this estimate as especially uncertain."
                )

            expected_home = float(estimates["expected_home_goals"])
            expected_away = float(estimates["expected_away_goals"])
            outcome_probs, scorelines = score_and_outcome_probabilities(
                expected_home, expected_away
            )
            predicted_result = max(outcome_probs, key=outcome_probs.get)

            st.divider()
            st.subheader(
                f"{home_team}  {expected_home:.2f} — {expected_away:.2f}  {away_team}"
            )
            st.caption(
                f"Most likely outcome: **{OUTCOME_LABELS[predicted_result]}** · "
                f"based on {estimates['home_matches']} earlier home-team matches "
                f"and {estimates['away_matches']} earlier away-team matches."
            )

            st.markdown("#### Outcome probabilities")
            probability_cols = st.columns(3)
            for column, outcome in zip(probability_cols, ("H", "D", "A")):
                with column:
                    st.metric(OUTCOME_LABELS[outcome], f"{outcome_probs[outcome]:.1%}")
                    st.progress(min(max(outcome_probs[outcome], 0.0), 1.0))

            scores_col, form_col = st.columns([1, 1.5])
            with scores_col:
                st.markdown("#### Most likely scorelines")
                top_scores = pd.DataFrame(
                    [
                        {
                            "Score": row["score"],
                            "Probability": f"{float(row['probability']):.1%}",
                        }
                        for row in scorelines[:5]
                    ]
                )
                st.dataframe(top_scores, hide_index=True, width="stretch")
            with form_col:
                st.markdown("#### Recent form")
                form_home, form_away = st.columns(2)
                with form_home:
                    render_recent_form(home_team, matches, as_of)
                with form_away:
                    render_recent_form(away_team, matches, as_of)

            st.info(
                "This is a simple statistical estimate, not a guarantee or betting "
                "advice. It does not account for lineups, injuries, odds, or live "
                "match events."
            )

with fixtures_tab:
    st.subheader("Upcoming fixture predictions")
    st.caption(
        f"Team statistics come only from {dataset_name} ({len(matches):,} completed "
        f"results through {latest_data_date:%d %b %Y}). For each fixture, results "
        "on that date and later dates are excluded. Fixture rows are never used as "
        "historical results."
    )

    if fixture_upload_error is not None:
        st.error(f"Fix the fixtures CSV and upload it again: {fixture_upload_error}")
    elif fixtures is None:
        st.info(
            "Upload a fixtures CSV in the sidebar with Date, HomeTeam and AwayTeam "
            "columns to create a batch forecast."
        )
    else:
        predictions = predict_fixtures(fixtures, matches)
        predicted_count = int((predictions["Prediction status"] == "Predicted").sum())
        if predicted_count:
            st.success(
                f"Predicted {predicted_count} of {len(predictions)} fixture(s)."
            )
        else:
            st.warning(
                "No fixtures had enough earlier venue-specific history to predict. "
                "Unscorable rows are left blank rather than estimated."
            )

        not_predicted_count = len(predictions) - predicted_count
        if predicted_count and not_predicted_count:
            st.warning(
                f"{not_predicted_count} fixture(s) were not predicted because a team "
                "was not present in earlier results or lacked enough earlier history. "
                "See the Prediction status column."
            )

        display_predictions = predictions.copy()
        display_predictions["Fixture date"] = display_predictions[
            "Fixture date"
        ].dt.strftime("%d %b %Y")
        st.dataframe(
            display_predictions,
            hide_index=True,
            width="stretch",
            column_config={
                "Expected home goals": st.column_config.NumberColumn(format="%.2f"),
                "Expected away goals": st.column_config.NumberColumn(format="%.2f"),
                "Home-win probability": st.column_config.NumberColumn(
                    label="Home-win probability", format="percent"
                ),
                "Draw probability": st.column_config.NumberColumn(
                    label="Draw probability", format="percent"
                ),
                "Away-win probability": st.column_config.NumberColumn(
                    label="Away-win probability", format="percent"
                ),
            },
        )
        st.caption(
            "The CSV stores probabilities as decimal values from 0 to 1 "
            "(for example, 0.62 means 62%)."
        )
        st.download_button(
            "Download fixture predictions as CSV",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="football_fixture_predictions.csv",
            mime="text/csv",
            key="download_fixture_predictions",
        )

with evaluation_tab:
    st.subheader("Historical model check")
    st.write(
        "A rolling backtest forecasts each match using only results from earlier "
        "calendar dates. The baseline uses the earlier league outcome frequencies."
    )
    results, metrics = cached_backtest(matches)
    if metrics is None:
        st.info(
            "This dataset does not contain enough history to run the backtest yet. "
            "A larger results file may provide a useful evaluation."
        )
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Matches tested", f"{metrics['matches_tested']:,}")
        metric_cols[1].metric("Outcome accuracy", f"{metrics['accuracy_pct']:.1f}%")
        metric_cols[2].metric("Model Brier score", f"{metrics['model_brier']:.3f}")
        metric_cols[3].metric("History baseline", f"{metrics['baseline_brier']:.3f}")

        if metrics["improvement_pct"] >= 0:
            st.caption(
                f"The model's Brier score was {metrics['improvement_pct']:.1f}% "
                "lower than the historical-frequency baseline in this backtest. "
                "Lower Brier scores are better."
            )
        else:
            st.caption(
                f"The model's Brier score was {abs(metrics['improvement_pct']):.1f}% "
                "higher than the historical-frequency baseline in this backtest. "
                "Lower Brier scores are better."
            )

        display_columns = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "Actual_Result",
            "Predicted_Result",
            "Home_Win_%",
            "Draw_%",
            "Away_Win_%",
        ]
        recent_backtest = results[display_columns].tail(20).copy()
        recent_backtest["Date"] = recent_backtest["Date"].dt.strftime("%d %b %Y")
        for column in ("Home_Win_%", "Draw_%", "Away_Win_%"):
            recent_backtest[column] = recent_backtest[column].map(
                lambda value: f"{value:.1f}%"
            )
        st.markdown("#### Latest backtest predictions")
        st.dataframe(
            recent_backtest,
            hide_index=True,
            width="stretch",
            column_config={
                "Actual_Result": st.column_config.TextColumn("Actual"),
                "Predicted_Result": st.column_config.TextColumn("Predicted"),
                "Home_Win_%": st.column_config.TextColumn("Home win"),
                "Draw_%": st.column_config.TextColumn("Draw"),
                "Away_Win_%": st.column_config.TextColumn("Away win"),
            },
        )
        st.caption(
            "A historical backtest is not a promise of future accuracy. The model "
            "uses recent results and venue averages, not player or market data."
        )

with data_tab:
    st.subheader("Dataset")
    info_cols = st.columns(3)
    info_cols[0].metric("Dataset", dataset_name)
    info_cols[1].metric("Completed matches", f"{len(matches):,}")
    info_cols[2].metric(
        "Teams",
        f"{len(set(matches['HomeTeam']) | set(matches['AwayTeam']))}",
    )
    st.caption(
        f"Available results run from {matches['Date'].min():%d %b %Y} to "
        f"{latest_data_date:%d %b %Y}. Current forecasts use only games before "
        f"{prediction_date:%d %b %Y}."
    )
    st.markdown(
        f"Bundled sample source: [football-data.co.uk]({DATA_SOURCE_URL}). "
        "The example file is the 2024–25 English Premier League season."
    )
    st.markdown("#### Recent results")
    preview = matches.tail(12).copy()
    preview["Date"] = preview["Date"].dt.strftime("%d %b %Y")
    preview["Score"] = preview["FTHG"].astype(str) + "–" + preview["FTAG"].astype(str)
    st.dataframe(
        preview[["Date", "HomeTeam", "Score", "AwayTeam", "FTR"]],
        hide_index=True,
        width="stretch",
        column_config={"FTR": st.column_config.TextColumn("Result (H/D/A)")},
    )

    if NOTEBOOK_PATH.exists():
        st.download_button(
            "Download the original notebook",
            data=NOTEBOOK_PATH.read_bytes(),
            file_name="Football_prediction.ipynb",
            mime="application/x-ipynb+json",
        )

