"""Context features and walk-forward draw-aware probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from model import (
    MIN_TEAM_MATCHES,
    OUTCOME_LABELS,
    OUTCOME_ORDER,
    MatchDataError,
    _brier_score,
    predict_venue_goals,
    score_and_outcome_probabilities,
    team_history_counts,
)

RECENT_MATCH_LIMIT = 10
MIN_VENUE_FORM_MATCHES = 3
HEAD_TO_HEAD_HALF_LIFE_DAYS = 3 * 365.25
MIN_TRAINING_EXAMPLES = 100
DEFAULT_MINIMUM_HISTORY = 50
REGULARIZATION_STRENGTH = 0.25

CONTEXT_FEATURE_NAMES = (
    "home_recent_points_per_game",
    "away_recent_points_per_game",
    "home_recent_draw_rate",
    "away_recent_draw_rate",
    "home_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_for",
    "away_recent_goals_against",
    "home_recent_goal_difference",
    "away_recent_goal_difference",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_home_goals",
    "h2h_away_goals",
    "h2h_log_weighted_meetings",
    "h2h_available",
)

FIXTURE_BATCH_COLUMNS = (
    "Fixture date",
    "Home team",
    "Away team",
    "Expected home goals",
    "Expected away goals",
    "Home-win probability",
    "Draw probability",
    "Away-win probability",
    "Poisson draw probability",
    "Draw adjustment (percentage points)",
    "Most likely outcome",
    "Prediction status",
)


def _empty_form_summary() -> dict[str, Any]:
    return {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "draw_rate": 0.0,
        "points_per_game": 0.0,
        "avg_goals_scored": 0.0,
        "avg_goals_conceded": 0.0,
        "goal_difference": 0.0,
    }


def _summarize_team_rows(team_rows: pd.DataFrame) -> dict[str, Any]:
    if team_rows.empty:
        return _empty_form_summary()

    at_home = team_rows["HomeTeam"].eq(team_rows["_Team"])
    team_rows = team_rows.copy()
    team_rows["_Scored"] = np.where(
        at_home, team_rows["FTHG"], team_rows["FTAG"]
    )
    team_rows["_Conceded"] = np.where(
        at_home, team_rows["FTAG"], team_rows["FTHG"]
    )
    wins = int((team_rows["_Scored"] > team_rows["_Conceded"]).sum())
    draws = int((team_rows["_Scored"] == team_rows["_Conceded"]).sum())
    losses = len(team_rows) - wins - draws
    count = len(team_rows)
    goals_scored = float(team_rows["_Scored"].mean())
    goals_conceded = float(team_rows["_Conceded"].mean())
    return {
        "matches": count,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "draw_rate": draws / count,
        "points_per_game": (3 * wins + draws) / count,
        "avg_goals_scored": goals_scored,
        "avg_goals_conceded": goals_conceded,
        "goal_difference": goals_scored - goals_conceded,
    }


def recent_form_summary(
    team: str,
    matches: pd.DataFrame,
    before: pd.Timestamp,
    limit: int = RECENT_MATCH_LIMIT,
) -> dict[str, Any]:
    """Summarize the team's last matches before a fixture, with venue splits."""
    team_rows = matches[
        (matches["Date"] < before)
        & ((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team))
    ].sort_values("Date", kind="stable").tail(limit).copy()
    team_rows["_Team"] = team

    home_rows = team_rows[team_rows["HomeTeam"] == team]
    away_rows = team_rows[team_rows["AwayTeam"] == team]
    return {
        **_summarize_team_rows(team_rows),
        "home": _summarize_team_rows(home_rows),
        "away": _summarize_team_rows(away_rows),
    }


def head_to_head_summary(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame,
    before: pd.Timestamp,
) -> dict[str, Any]:
    """Return prior meetings from the selected fixture's perspective."""
    earlier = matches[matches["Date"] < before]
    meetings = earlier[
        (
            (earlier["HomeTeam"] == home_team)
            & (earlier["AwayTeam"] == away_team)
        )
        | (
            (earlier["HomeTeam"] == away_team)
            & (earlier["AwayTeam"] == home_team)
        )
    ].sort_values("Date", kind="stable")

    weighted_outcomes = {outcome: 0.0 for outcome in OUTCOME_ORDER}
    weighted_home_goals = 0.0
    weighted_away_goals = 0.0
    weight_sum = 0.0
    for _, meeting in meetings.iterrows():
        if meeting["HomeTeam"] == home_team:
            home_goals = int(meeting["FTHG"])
            away_goals = int(meeting["FTAG"])
        else:
            home_goals = int(meeting["FTAG"])
            away_goals = int(meeting["FTHG"])

        outcome = (
            "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        )
        age_days = max(0, (before - meeting["Date"]).days)
        weight = 0.5 ** (age_days / HEAD_TO_HEAD_HALF_LIFE_DAYS)
        weighted_outcomes[outcome] += weight
        weighted_home_goals += home_goals * weight
        weighted_away_goals += away_goals * weight
        weight_sum += weight

    if not len(meetings) or weight_sum == 0:
        return {
            "meetings": 0,
            "weighted_meetings": 0.0,
            "home_win_rate": None,
            "draw_rate": None,
            "away_win_rate": None,
            "avg_home_goals": None,
            "avg_away_goals": None,
        }

    return {
        "meetings": len(meetings),
        "weighted_meetings": weight_sum,
        "home_win_rate": weighted_outcomes["H"] / weight_sum,
        "draw_rate": weighted_outcomes["D"] / weight_sum,
        "away_win_rate": weighted_outcomes["A"] / weight_sum,
        "avg_home_goals": weighted_home_goals / weight_sum,
        "avg_away_goals": weighted_away_goals / weight_sum,
    }


def build_prediction_context(
    home_team: str,
    away_team: str,
    history: pd.DataFrame,
    before: pd.Timestamp,
) -> dict[str, Any] | None:
    """Build baseline probabilities and pre-fixture form/head-to-head features."""
    estimates = predict_venue_goals(home_team, away_team, history)
    if estimates is None:
        return None
    baseline_probs, scorelines = score_and_outcome_probabilities(
        float(estimates["expected_home_goals"]),
        float(estimates["expected_away_goals"]),
    )
    home_form = recent_form_summary(home_team, history, before)
    away_form = recent_form_summary(away_team, history, before)
    h2h = head_to_head_summary(home_team, away_team, history, before)

    h2h_available = h2h["meetings"] > 0
    features = np.asarray(
        [
            home_form["points_per_game"],
            away_form["points_per_game"],
            home_form["draw_rate"],
            away_form["draw_rate"],
            home_form["avg_goals_scored"],
            home_form["avg_goals_conceded"],
            away_form["avg_goals_scored"],
            away_form["avg_goals_conceded"],
            home_form["goal_difference"],
            away_form["goal_difference"],
            h2h["home_win_rate"] or 0.0,
            h2h["draw_rate"] or 0.0,
            h2h["away_win_rate"] or 0.0,
            h2h["avg_home_goals"] or 0.0,
            h2h["avg_away_goals"] or 0.0,
            math.log1p(h2h["weighted_meetings"]),
            float(h2h_available),
        ],
        dtype=float,
    )
    return {
        "estimates": estimates,
        "baseline_probabilities": baseline_probs,
        "scorelines": scorelines,
        "features": features,
        "home_form": home_form,
        "away_form": away_form,
        "head_to_head": h2h,
    }


def season_start_year(value: pd.Timestamp) -> int:
    """Infer an English football season beginning on 1 August."""
    timestamp = pd.Timestamp(value)
    return timestamp.year if timestamp.month >= 8 else timestamp.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


@dataclass(frozen=True)
class ResidualOutcomeModel:
    """Regularized multinomial calibration applied as a residual to Poisson."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    training_examples: int
    training_seasons: tuple[str, ...]

    def predict(
        self, features: np.ndarray, baseline_probabilities: dict[str, float]
    ) -> dict[str, float]:
        standardized = (features - self.feature_mean) / self.feature_scale
        offset = np.log(
            np.clip(
                np.asarray(
                    [baseline_probabilities[outcome] for outcome in OUTCOME_ORDER],
                    dtype=float,
                ),
                1e-9,
                1.0,
            )
        )
        logits = offset + self.intercept + standardized @ self.coefficients
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return {
            outcome: float(probabilities[index])
            for index, outcome in enumerate(OUTCOME_ORDER)
        }


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(stable)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _fit_residual_model(
    samples: list[dict[str, Any]],
    minimum_training_examples: int = MIN_TRAINING_EXAMPLES,
) -> ResidualOutcomeModel | None:
    if len(samples) < minimum_training_examples:
        return None
    labels = [str(sample["actual"]) for sample in samples]
    if set(labels) != set(OUTCOME_ORDER):
        return None

    features = np.vstack([sample["features"] for sample in samples]).astype(float)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale

    baselines = np.asarray(
        [
            [sample["baseline"][outcome] for outcome in OUTCOME_ORDER]
            for sample in samples
        ],
        dtype=float,
    )
    log_baselines = np.log(np.clip(baselines, 1e-9, 1.0))
    label_indexes = np.asarray(
        [OUTCOME_ORDER.index(label) for label in labels], dtype=int
    )
    actual = np.eye(len(OUTCOME_ORDER), dtype=float)[label_indexes]
    coefficients = np.zeros(
        (standardized.shape[1], len(OUTCOME_ORDER)), dtype=float
    )
    intercept = np.zeros(len(OUTCOME_ORDER), dtype=float)

    # The fixed L2 penalty shrinks learned adjustments toward the existing model.
    # It is not tuned on the season being evaluated.
    learning_rate = 0.08
    for _ in range(600):
        probabilities = _softmax_rows(
            log_baselines + intercept + standardized @ coefficients
        )
        errors = probabilities - actual
        gradient_coefficients = (
            standardized.T @ errors / len(samples)
            + REGULARIZATION_STRENGTH * coefficients
        )
        gradient_intercept = errors.mean(axis=0)
        coefficients -= learning_rate * gradient_coefficients
        intercept -= learning_rate * gradient_intercept
        if max(
            float(np.max(np.abs(gradient_coefficients))),
            float(np.max(np.abs(gradient_intercept))),
        ) < 1e-6:
            break

    training_seasons = tuple(
        sorted({str(sample["season"]) for sample in samples})
    )
    return ResidualOutcomeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        training_examples=len(samples),
        training_seasons=training_seasons,
    )


def _create_training_examples(
    matches: pd.DataFrame,
    minimum_history: int = DEFAULT_MINIMUM_HISTORY,
    before: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    ordered = matches.sort_values("Date", kind="stable")
    if before is not None:
        ordered = ordered[ordered["Date"] < before]

    examples: list[dict[str, Any]] = []
    for _, match in ordered.iterrows():
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

        context = build_prediction_context(
            home_team, away_team, history, pd.Timestamp(match["Date"])
        )
        if context is None:
            continue

        outcome_counts = history["FTR"].value_counts()
        denominator = len(history) + len(OUTCOME_ORDER)
        history_frequency = {
            outcome: (int(outcome_counts.get(outcome, 0)) + 1) / denominator
            for outcome in OUTCOME_ORDER
        }
        start_year = season_start_year(pd.Timestamp(match["Date"]))
        examples.append(
            {
                "Date": pd.Timestamp(match["Date"]),
                "season_year": start_year,
                "season": season_label(start_year),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "actual": str(match["FTR"]),
                "features": context["features"],
                "baseline": context["baseline_probabilities"],
                "history_frequency": history_frequency,
            }
        )
    return examples


def fit_enhanced_model_as_of(
    matches: pd.DataFrame,
    before: pd.Timestamp,
    minimum_history: int = DEFAULT_MINIMUM_HISTORY,
    minimum_training_examples: int = MIN_TRAINING_EXAMPLES,
) -> ResidualOutcomeModel | None:
    """Fit only from match labels already known before the requested date."""
    samples = _create_training_examples(
        matches, minimum_history=minimum_history, before=before
    )
    return _fit_residual_model(samples, minimum_training_examples)


def _performance_metrics(
    actual: list[str], probabilities: list[dict[str, float]]
) -> dict[str, float | int]:
    if not actual:
        return {
            "matches": 0,
            "accuracy_pct": 0.0,
            "brier_score": 0.0,
            "draw_precision_pct": 0.0,
            "draw_recall_pct": 0.0,
            "draw_f1_pct": 0.0,
            "mean_predicted_draw_pct": 0.0,
            "actual_draw_frequency_pct": 0.0,
        }

    predicted = [max(row, key=row.get) for row in probabilities]
    true_draws = sum(value == "D" for value in actual)
    predicted_draws = sum(value == "D" for value in predicted)
    true_positive_draws = sum(
        truth == "D" and guess == "D" for truth, guess in zip(actual, predicted)
    )
    draw_precision = (
        true_positive_draws / predicted_draws if predicted_draws else 0.0
    )
    draw_recall = true_positive_draws / true_draws if true_draws else 0.0
    draw_f1 = (
        2 * draw_precision * draw_recall / (draw_precision + draw_recall)
        if draw_precision + draw_recall
        else 0.0
    )
    return {
        "matches": len(actual),
        "accuracy_pct": sum(truth == guess for truth, guess in zip(actual, predicted))
        / len(actual)
        * 100,
        "brier_score": sum(
            _brier_score(probability, truth)
            for truth, probability in zip(actual, probabilities)
        )
        / len(actual),
        "draw_precision_pct": draw_precision * 100,
        "draw_recall_pct": draw_recall * 100,
        "draw_f1_pct": draw_f1 * 100,
        "mean_predicted_draw_pct": (
            sum(row["D"] for row in probabilities) / len(probabilities) * 100
        ),
        "actual_draw_frequency_pct": true_draws / len(actual) * 100,
    }


def _metric_row(
    season: str,
    model_name: str,
    actual: list[str],
    probabilities: list[dict[str, float]],
    training_examples: int = 0,
) -> dict[str, Any]:
    return {
        "Season": season,
        "Model": model_name,
        **_performance_metrics(actual, probabilities),
        "Training examples": training_examples,
    }


def run_comparative_backtest(
    matches: pd.DataFrame,
    minimum_history: int = DEFAULT_MINIMUM_HISTORY,
    minimum_training_examples: int = MIN_TRAINING_EXAMPLES,
) -> tuple[pd.DataFrame, dict[str, Any] | None, pd.DataFrame | None]:
    """Compare Poisson and enhanced predictions, training only on earlier seasons."""
    examples = _create_training_examples(matches, minimum_history=minimum_history)
    if not examples:
        return pd.DataFrame(), None, None

    rows: list[dict[str, Any]] = []
    season_metric_rows: list[dict[str, Any]] = []
    for target_year in sorted({int(example["season_year"]) for example in examples}):
        training = [
            example for example in examples if int(example["season_year"]) < target_year
        ]
        enhanced_model = _fit_residual_model(
            training, minimum_training_examples=minimum_training_examples
        )
        target_examples = [
            example
            for example in examples
            if int(example["season_year"]) == target_year
        ]
        season = season_label(target_year)
        season_actual: list[str] = []
        season_poisson: list[dict[str, float]] = []
        season_enhanced: list[dict[str, float]] = []
        season_frequency: list[dict[str, float]] = []

        for example in target_examples:
            baseline = example["baseline"]
            enhanced = (
                enhanced_model.predict(example["features"], baseline)
                if enhanced_model is not None
                else dict(baseline)
            )
            history_frequency = example["history_frequency"]
            actual_result = str(example["actual"])
            season_actual.append(actual_result)
            season_poisson.append(baseline)
            season_enhanced.append(enhanced)
            season_frequency.append(history_frequency)
            rows.append(
                {
                    "Date": example["Date"],
                    "Season": season,
                    "HomeTeam": example["HomeTeam"],
                    "AwayTeam": example["AwayTeam"],
                    "Actual_Result": actual_result,
                    "Poisson_Result": max(baseline, key=baseline.get),
                    "Poisson_Home_Win_%": baseline["H"] * 100,
                    "Poisson_Draw_%": baseline["D"] * 100,
                    "Poisson_Away_Win_%": baseline["A"] * 100,
                    "Poisson_Brier": _brier_score(baseline, actual_result),
                    "Enhanced_Result": max(enhanced, key=enhanced.get),
                    "Enhanced_Home_Win_%": enhanced["H"] * 100,
                    "Enhanced_Draw_%": enhanced["D"] * 100,
                    "Enhanced_Away_Win_%": enhanced["A"] * 100,
                    "Enhanced_Brier": _brier_score(enhanced, actual_result),
                    "Enhanced_Model_Used": enhanced_model is not None,
                    "Enhanced_Training_Matches": (
                        enhanced_model.training_examples
                        if enhanced_model is not None
                        else 0
                    ),
                    "History_Frequency_Home_Win_%": history_frequency["H"] * 100,
                    "History_Frequency_Draw_%": history_frequency["D"] * 100,
                    "History_Frequency_Away_Win_%": history_frequency["A"] * 100,
                    "History_Frequency_Brier": _brier_score(
                        history_frequency, actual_result
                    ),
                }
            )

        training_count = (
            enhanced_model.training_examples if enhanced_model is not None else 0
        )
        season_metric_rows.extend(
            [
                _metric_row(
                    season, "Existing venue-aware Poisson", season_actual,
                    season_poisson, training_count,
                ),
                _metric_row(
                    season, "Enhanced draw-aware model", season_actual,
                    season_enhanced, training_count,
                ),
                _metric_row(
                    season, "Historical outcome-frequency reference", season_actual,
                    season_frequency, 0,
                ),
            ]
        )

    results = pd.DataFrame(rows).sort_values(
        ["Date", "HomeTeam", "AwayTeam"], kind="stable"
    ).reset_index(drop=True)
    actual = results["Actual_Result"].astype(str).tolist()

    def probability_rows(prefix: str) -> list[dict[str, float]]:
        return [
            {
                "H": float(row[f"{prefix}_Home_Win_%"]) / 100,
                "D": float(row[f"{prefix}_Draw_%"]) / 100,
                "A": float(row[f"{prefix}_Away_Win_%"]) / 100,
            }
            for _, row in results.iterrows()
        ]

    history_probability_rows = [
        {
            "H": float(row["History_Frequency_Home_Win_%"]) / 100,
            "D": float(row["History_Frequency_Draw_%"]) / 100,
            "A": float(row["History_Frequency_Away_Win_%"]) / 100,
        }
        for _, row in results.iterrows()
    ]
    overall_metrics = {
        "matches_tested": len(results),
        "existing_poisson": _performance_metrics(
            actual, probability_rows("Poisson")
        ),
        "enhanced": _performance_metrics(
            actual, probability_rows("Enhanced")
        ),
        "history_frequency": _performance_metrics(actual, history_probability_rows),
        "enhanced_training_matches": int(results["Enhanced_Model_Used"].sum()),
        "enhanced_fallback_matches": int((~results["Enhanced_Model_Used"]).sum()),
    }
    return results, overall_metrics, pd.DataFrame(season_metric_rows)


def predict_fixture_batch(
    fixtures: pd.DataFrame,
    historical_matches: pd.DataFrame,
    minimum_history: int = DEFAULT_MINIMUM_HISTORY,
    minimum_training_examples: int = MIN_TRAINING_EXAMPLES,
) -> pd.DataFrame:
    """Predict fixture rows from earlier results, with a Poisson fallback."""
    predictions: list[dict[str, Any]] = []
    for _, fixture in fixtures.iterrows():
        fixture_date = pd.Timestamp(fixture["Date"]).normalize()
        home_team = str(fixture["HomeTeam"])
        away_team = str(fixture["AwayTeam"])
        history = historical_matches[
            historical_matches["Date"] < fixture_date
        ].copy()

        prediction: dict[str, Any] = {
            "Fixture date": fixture_date,
            "Home team": home_team,
            "Away team": away_team,
            "Expected home goals": float("nan"),
            "Expected away goals": float("nan"),
            "Home-win probability": float("nan"),
            "Draw probability": float("nan"),
            "Away-win probability": float("nan"),
            "Poisson draw probability": float("nan"),
            "Draw adjustment (percentage points)": float("nan"),
            "Most likely outcome": "",
            "Prediction status": "",
        }

        present_teams = set(history["HomeTeam"]) | set(history["AwayTeam"])
        missing_teams = [
            team for team in (home_team, away_team) if team not in present_teams
        ]
        if missing_teams:
            prediction["Prediction status"] = (
                "Not predicted: no earlier results for "
                + ", ".join(missing_teams)
                + "; team names must match the results CSV exactly."
            )
            predictions.append(prediction)
            continue

        insufficient: list[str] = []
        for team in (home_team, away_team):
            counts = team_history_counts(team, history)
            missing_venues = [
                venue
                for venue, count in (
                    ("home", counts["home_matches"]),
                    ("away", counts["away_matches"]),
                )
                if count == 0
            ]
            if counts["total_matches"] < MIN_TEAM_MATCHES or missing_venues:
                detail = f"{team}: {counts['total_matches']}/{MIN_TEAM_MATCHES} matches"
                if missing_venues:
                    detail += f", no earlier {' and '.join(missing_venues)} results"
                insufficient.append(detail)

        if insufficient:
            prediction["Prediction status"] = (
                "Not predicted: insufficient earlier history for "
                + "; ".join(insufficient)
                + "."
            )
            predictions.append(prediction)
            continue

        context = build_prediction_context(
            home_team, away_team, history, fixture_date
        )
        if context is None:
            prediction["Prediction status"] = (
                "Not predicted: both teams need earlier home and away results."
            )
            predictions.append(prediction)
            continue

        baseline = context["baseline_probabilities"]
        enhanced_model = fit_enhanced_model_as_of(
            historical_matches,
            fixture_date,
            minimum_history=minimum_history,
            minimum_training_examples=minimum_training_examples,
        )
        final_probabilities = (
            enhanced_model.predict(context["features"], baseline)
            if enhanced_model is not None
            else baseline
        )
        most_likely = max(final_probabilities, key=final_probabilities.get)
        prediction.update(
            {
                "Expected home goals": float(
                    context["estimates"]["expected_home_goals"]
                ),
                "Expected away goals": float(
                    context["estimates"]["expected_away_goals"]
                ),
                "Home-win probability": final_probabilities["H"],
                "Draw probability": final_probabilities["D"],
                "Away-win probability": final_probabilities["A"],
                "Poisson draw probability": baseline["D"],
                "Draw adjustment (percentage points)": (
                    final_probabilities["D"] - baseline["D"]
                )
                * 100,
                "Most likely outcome": OUTCOME_LABELS[most_likely],
                "Prediction status": (
                    "Predicted with enhanced model"
                    if enhanced_model is not None
                    else "Predicted with Poisson fallback: not enough earlier "
                    "training examples for enhancement"
                ),
            }
        )
        predictions.append(prediction)

    return pd.DataFrame(predictions, columns=FIXTURE_BATCH_COLUMNS)
