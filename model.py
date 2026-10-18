"""Football match forecasting functions adapted from the supplied notebook."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

OUTCOME_ORDER = ("H", "D", "A")
OUTCOME_LABELS = {
    "H": "Home win",
    "D": "Draw",
    "A": "Away win",
}
REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
FIXTURE_REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam")
FIXTURE_PREDICTION_COLUMNS = (
    "Fixture date",
    "Home team",
    "Away team",
    "Expected home goals",
    "Expected away goals",
    "Home-win probability",
    "Draw probability",
    "Away-win probability",
    "Most likely outcome",
    "Prediction status",
)
MIN_TEAM_MATCHES = 5


class MatchDataError(ValueError):
    """Raised when a match-results CSV cannot be used by the model."""


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates as year-first and other common football dates as day-first."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    iso_dates = text.str.match(
        r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$", na=False
    )
    year_first_slash_dates = text.str.match(
        r"^\d{4}/\d{1,2}/\d{1,2}(?:\s.*)?$", na=False
    )
    day_first_dates = ~(iso_dates | year_first_slash_dates)

    if iso_dates.any():
        parsed.loc[iso_dates] = pd.to_datetime(
            text.loc[iso_dates], format="ISO8601", errors="coerce"
        )
    if year_first_slash_dates.any():
        parsed.loc[year_first_slash_dates] = pd.to_datetime(
            text.loc[year_first_slash_dates],
            format="mixed",
            dayfirst=False,
            errors="coerce",
        )
    if day_first_dates.any():
        parsed.loc[day_first_dates] = pd.to_datetime(
            text.loc[day_first_dates],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    return parsed


def prepare_fixtures(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an upcoming-fixtures table without treating it as match results."""
    missing = [column for column in FIXTURE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MatchDataError(
            "The fixtures CSV is missing required columns: " + ", ".join(missing)
        )

    fixtures = frame.loc[:, FIXTURE_REQUIRED_COLUMNS].copy().reset_index(drop=True)
    fixtures["Date"] = _parse_dates(fixtures["Date"]).dt.normalize()
    fixtures["HomeTeam"] = fixtures["HomeTeam"].astype("string").str.strip()
    fixtures["AwayTeam"] = fixtures["AwayTeam"].astype("string").str.strip()

    invalid_rows: list[str] = []
    for row_index, row in fixtures.iterrows():
        problems: list[str] = []
        if pd.isna(row["Date"]):
            problems.append("missing or invalid Date")
        if pd.isna(row["HomeTeam"]) or not row["HomeTeam"]:
            problems.append("missing HomeTeam")
        if pd.isna(row["AwayTeam"]) or not row["AwayTeam"]:
            problems.append("missing AwayTeam")
        if (
            pd.notna(row["HomeTeam"])
            and pd.notna(row["AwayTeam"])
            and row["HomeTeam"]
            and row["HomeTeam"] == row["AwayTeam"]
        ):
            problems.append("home and away teams must differ")
        if problems:
            invalid_rows.append(f"CSV row {row_index + 2}: {', '.join(problems)}")

    if invalid_rows:
        details = "; ".join(invalid_rows[:10])
        if len(invalid_rows) > 10:
            details += f"; and {len(invalid_rows) - 10} more invalid row(s)"
        raise MatchDataError("Invalid fixture data: " + details)
    if fixtures.empty:
        raise MatchDataError("No fixtures were found in the CSV.")

    duplicate_mask = fixtures.duplicated(
        subset=["Date", "HomeTeam", "AwayTeam"], keep=False
    )
    if duplicate_mask.any():
        duplicate_rows = ", ".join(
            str(row_index + 2) for row_index in fixtures.index[duplicate_mask]
        )
        raise MatchDataError(
            "Duplicate fixtures found on CSV row(s) "
            f"{duplicate_rows} (same date, home team, and away team)."
        )

    return fixtures


def prepare_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize a football-data.co.uk style results table."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MatchDataError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    matches = frame.loc[:, REQUIRED_COLUMNS].copy()
    matches["Date"] = _parse_dates(matches["Date"])
    matches["FTHG"] = pd.to_numeric(matches["FTHG"], errors="coerce")
    matches["FTAG"] = pd.to_numeric(matches["FTAG"], errors="coerce")
    matches["HomeTeam"] = matches["HomeTeam"].astype("string").str.strip()
    matches["AwayTeam"] = matches["AwayTeam"].astype("string").str.strip()
    matches = matches.dropna(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    matches = matches[
        (matches["HomeTeam"] != "")
        & (matches["AwayTeam"] != "")
        & (matches["FTHG"] >= 0)
        & (matches["FTAG"] >= 0)
    ].copy()

    if matches.empty:
        raise MatchDataError("No completed matches with valid scores were found.")

    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)
    matches["FTR"] = "D"
    matches.loc[matches["FTHG"] > matches["FTAG"], "FTR"] = "H"
    matches.loc[matches["FTHG"] < matches["FTAG"], "FTR"] = "A"
    return matches.sort_values("Date").reset_index(drop=True)


def predict_fixtures(
    fixtures: pd.DataFrame, historical_matches: pd.DataFrame
) -> pd.DataFrame:
    """Forecast fixtures from only earlier completed results; leave unsupported rows blank."""
    predictions: list[dict[str, object]] = []

    for _, fixture in fixtures.iterrows():
        fixture_date = fixture["Date"]
        home_team = str(fixture["HomeTeam"])
        away_team = str(fixture["AwayTeam"])
        past_matches = historical_matches[
            historical_matches["Date"] < fixture_date
        ].copy()
        teams_with_earlier_results = set(past_matches["HomeTeam"]) | set(
            past_matches["AwayTeam"]
        )
        unavailable_teams = [
            team
            for team in (home_team, away_team)
            if team not in teams_with_earlier_results
        ]

        prediction: dict[str, object] = {
            "Fixture date": fixture_date,
            "Home team": home_team,
            "Away team": away_team,
            "Expected home goals": float("nan"),
            "Expected away goals": float("nan"),
            "Home-win probability": float("nan"),
            "Draw probability": float("nan"),
            "Away-win probability": float("nan"),
            "Most likely outcome": "",
            "Prediction status": "",
        }

        if unavailable_teams:
            prediction["Prediction status"] = (
                "Not predicted: no earlier results for "
                + ", ".join(unavailable_teams)
                + "; team names must match the results CSV exactly."
            )
            predictions.append(prediction)
            continue

        insufficient_teams: list[str] = []
        for team in (home_team, away_team):
            counts = team_history_counts(team, past_matches)
            missing_venues = []
            if counts["home_matches"] == 0:
                missing_venues.append("home")
            if counts["away_matches"] == 0:
                missing_venues.append("away")
            if (
                counts["total_matches"] < MIN_TEAM_MATCHES
                or missing_venues
            ):
                reason = f"{team}: {counts['total_matches']}/{MIN_TEAM_MATCHES} matches"
                if missing_venues:
                    reason += f", no earlier {' and '.join(missing_venues)} results"
                insufficient_teams.append(reason)

        if insufficient_teams:
            prediction["Prediction status"] = (
                "Not predicted: insufficient earlier history for "
                + "; ".join(insufficient_teams)
                + "."
            )
            predictions.append(prediction)
            continue

        estimates = predict_venue_goals(home_team, away_team, past_matches)
        if estimates is None:
            prediction["Prediction status"] = (
                "Not predicted: both teams need earlier home and away results."
            )
            predictions.append(prediction)
            continue

        expected_home = float(estimates["expected_home_goals"])
        expected_away = float(estimates["expected_away_goals"])
        outcome_probs, _ = score_and_outcome_probabilities(
            expected_home, expected_away
        )
        most_likely = max(outcome_probs, key=outcome_probs.get)
        prediction.update(
            {
                "Expected home goals": expected_home,
                "Expected away goals": expected_away,
                "Home-win probability": outcome_probs["H"],
                "Draw probability": outcome_probs["D"],
                "Away-win probability": outcome_probs["A"],
                "Most likely outcome": OUTCOME_LABELS[most_likely],
                "Prediction status": "Predicted",
            }
        )
        predictions.append(prediction)

    return pd.DataFrame(predictions, columns=FIXTURE_PREDICTION_COLUMNS)


def team_history_counts(team: str, matches: pd.DataFrame) -> dict[str, int]:
    home_games = int((matches["HomeTeam"] == team).sum())
    away_games = int((matches["AwayTeam"] == team).sum())
    return {
        "home_matches": home_games,
        "away_matches": away_games,
        "total_matches": home_games + away_games,
    }


def get_venue_stats(team: str, matches: pd.DataFrame) -> dict[str, float | int]:
    home_matches = matches[matches["HomeTeam"] == team]
    away_matches = matches[matches["AwayTeam"] == team]
    return {
        "home_matches": len(home_matches),
        "home_scored": float(home_matches["FTHG"].mean()),
        "home_conceded": float(home_matches["FTAG"].mean()),
        "away_matches": len(away_matches),
        "away_scored": float(away_matches["FTAG"].mean()),
        "away_conceded": float(away_matches["FTHG"].mean()),
    }


def predict_venue_goals(
    home_team: str, away_team: str, past_matches: pd.DataFrame
) -> dict[str, float | int] | None:
    """Estimate goals using each side's venue-specific scoring and conceding."""
    if home_team == away_team:
        raise MatchDataError("Choose two different teams.")

    home = get_venue_stats(home_team, past_matches)
    away = get_venue_stats(away_team, past_matches)
    if (
        home["home_matches"] == 0
        or home["away_matches"] == 0
        or away["home_matches"] == 0
        or away["away_matches"] == 0
    ):
        return None

    expected_home = (home["home_scored"] + away["away_conceded"]) / 2
    expected_away = (away["away_scored"] + home["home_conceded"]) / 2
    return {
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "home_matches": int(home["home_matches"] + home["away_matches"]),
        "away_matches": int(away["home_matches"] + away["away_matches"]),
    }


def _poisson_probability(expected_goals: float, actual_goals: int) -> float:
    if expected_goals < 0 or not math.isfinite(expected_goals):
        raise MatchDataError("Expected goals must be a finite, non-negative number.")
    if expected_goals == 0:
        return 1.0 if actual_goals == 0 else 0.0
    return math.exp(
        -expected_goals
        + actual_goals * math.log(expected_goals)
        - math.lgamma(actual_goals + 1)
    )


def score_and_outcome_probabilities(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int | None = None,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Return normalized 1X2 probabilities and the scoreline probability grid."""
    if expected_home_goals < 0 or expected_away_goals < 0:
        raise MatchDataError("Expected goals cannot be negative.")

    if max_goals is None:
        highest_expectation = max(expected_home_goals, expected_away_goals)
        max_goals = min(
            20,
            max(10, math.ceil(highest_expectation + 7 * math.sqrt(highest_expectation + 1))),
        )

    scorelines: list[dict[str, float | str]] = []
    for home_goals in range(max_goals + 1):
        home_probability = _poisson_probability(expected_home_goals, home_goals)
        for away_goals in range(max_goals + 1):
            probability = home_probability * _poisson_probability(
                expected_away_goals, away_goals
            )
            scorelines.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": probability,
                }
            )

    total_probability = sum(float(row["probability"]) for row in scorelines)
    if total_probability <= 0:
        raise MatchDataError("Could not calculate scoreline probabilities.")

    for row in scorelines:
        row["probability"] = float(row["probability"]) / total_probability

    outcomes = {"H": 0.0, "D": 0.0, "A": 0.0}
    for row in scorelines:
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        outcomes[result] += float(row["probability"])

    return outcomes, sorted(
        scorelines, key=lambda row: float(row["probability"]), reverse=True
    )


def recent_team_matches(
    team: str, matches: pd.DataFrame, before: pd.Timestamp, limit: int = 5
) -> pd.DataFrame:
    """Return the team's latest results before a prediction date."""
    eligible = matches[
        (matches["Date"] < before)
        & ((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team))
    ].tail(limit)

    rows: list[dict[str, object]] = []
    for _, match in eligible.iterrows():
        at_home = match["HomeTeam"] == team
        scored = int(match["FTHG"] if at_home else match["FTAG"])
        conceded = int(match["FTAG"] if at_home else match["FTHG"])
        result = "W" if scored > conceded else "L" if scored < conceded else "D"
        rows.append(
            {
                "Date": match["Date"].strftime("%d %b %Y"),
                "Opponent": match["AwayTeam"] if at_home else match["HomeTeam"],
                "Venue": "Home" if at_home else "Away",
                "Score": f"{scored}-{conceded}",
                "Result": result,
            }
        )
    return pd.DataFrame(rows)


def _brier_score(
    probabilities: Mapping[str, float], actual_result: str
) -> float:
    return sum(
        (probabilities[outcome] - float(actual_result == outcome)) ** 2
        for outcome in OUTCOME_ORDER
    )


def run_backtest(
    matches: pd.DataFrame, minimum_history: int = 50
) -> tuple[pd.DataFrame, dict[str, float | int] | None]:
    """Evaluate chronologically, using only matches on earlier dates."""
    predictions: list[dict[str, object]] = []
    for _, match in matches.iterrows():
        history = matches[matches["Date"] < match["Date"]]
        if len(history) < minimum_history:
            continue

        home_team = str(match["HomeTeam"])
        away_team = str(match["AwayTeam"])
        home_counts = team_history_counts(home_team, history)
        away_counts = team_history_counts(away_team, history)
        if (
            home_counts["total_matches"] < MIN_TEAM_MATCHES
            or away_counts["total_matches"] < MIN_TEAM_MATCHES
        ):
            continue

        estimates = predict_venue_goals(home_team, away_team, history)
        if estimates is None:
            continue

        outcome_probs, _ = score_and_outcome_probabilities(
            float(estimates["expected_home_goals"]),
            float(estimates["expected_away_goals"]),
        )
        history_counts = history["FTR"].value_counts()
        denominator = len(history) + len(OUTCOME_ORDER)
        baseline_probs = {
            outcome: (int(history_counts.get(outcome, 0)) + 1) / denominator
            for outcome in OUTCOME_ORDER
        }
        actual_result = str(match["FTR"])
        predicted_result = max(outcome_probs, key=outcome_probs.get)
        predictions.append(
            {
                "Date": match["Date"],
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Actual_Result": actual_result,
                "Predicted_Result": predicted_result,
                "Home_Win_%": outcome_probs["H"] * 100,
                "Draw_%": outcome_probs["D"] * 100,
                "Away_Win_%": outcome_probs["A"] * 100,
                "Model_Brier": _brier_score(outcome_probs, actual_result),
                "Baseline_Brier": _brier_score(baseline_probs, actual_result),
            }
        )

    results = pd.DataFrame(predictions)
    if results.empty:
        return results, None

    model_brier = float(results["Model_Brier"].mean())
    baseline_brier = float(results["Baseline_Brier"].mean())
    metrics: dict[str, float | int] = {
        "matches_tested": len(results),
        "accuracy_pct": float(
            (results["Predicted_Result"] == results["Actual_Result"]).mean() * 100
        ),
        "model_brier": model_brier,
        "baseline_brier": baseline_brier,
        "improvement_pct": (
            (baseline_brier - model_brier) / baseline_brier * 100
            if baseline_brier
            else 0.0
        ),
    }
    return results, metrics
