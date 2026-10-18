from __future__ import annotations

import unittest
from io import StringIO

import pandas as pd

from model import (
    FIXTURE_PREDICTION_COLUMNS,
    MatchDataError,
    predict_fixtures,
    predict_venue_goals,
    prepare_fixtures,
    prepare_matches,
)


def make_history(match_count: int = 6) -> pd.DataFrame:
    start = pd.Timestamp("2026-08-01")
    rows: list[dict[str, object]] = []
    for offset in range(match_count):
        if offset % 2 == 0:
            home_team, away_team = "Alpha", "Beta"
            home_goals, away_goals = 2 + offset, 1
        else:
            home_team, away_team = "Beta", "Alpha"
            home_goals, away_goals = 1, 2 + offset
        rows.append(
            {
                "Date": start + pd.Timedelta(days=offset),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": "D",
            }
        )
    return prepare_matches(pd.DataFrame(rows))


class HistoricalResultsUploadTests(unittest.TestCase):
    def test_results_csv_still_uses_results_schema_and_recomputes_result(self) -> None:
        csv_text = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
            "01/08/2026,Alpha,Beta,2,0,A\n"
            "02/08/2026,Alpha,Beta,,,H\n"
        )
        matches = prepare_matches(pd.read_csv(StringIO(csv_text)))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]["FTR"], "H")
        self.assertEqual(matches.iloc[0]["FTHG"], 2)


class UpcomingFixturesTests(unittest.TestCase):
    def test_fixture_csv_needs_only_date_and_team_columns(self) -> None:
        csv_text = "Date,HomeTeam,AwayTeam,Competition\n2026-08-07,Alpha,Beta,EPL\n"
        fixtures = prepare_fixtures(pd.read_csv(StringIO(csv_text)))

        self.assertEqual(list(fixtures.columns), ["Date", "HomeTeam", "AwayTeam"])
        self.assertEqual(fixtures.iloc[0]["Date"], pd.Timestamp("2026-08-07"))

    def test_iso_and_day_first_dates_parse_correctly_in_the_same_file(self) -> None:
        frame = pd.DataFrame(
            [
                {"Date": "07/08/2026", "HomeTeam": "Alpha", "AwayTeam": "Beta"},
                {"Date": "2026-08-08", "HomeTeam": "Beta", "AwayTeam": "Alpha"},
            ]
        )

        fixtures = prepare_fixtures(frame)

        self.assertEqual(
            fixtures["Date"].tolist(),
            [pd.Timestamp("2026-08-07"), pd.Timestamp("2026-08-08")],
        )

    def test_fixture_validation_reports_invalid_dates_and_missing_teams(self) -> None:
        frame = pd.DataFrame(
            [
                {"Date": "not-a-date", "HomeTeam": "Alpha", "AwayTeam": "Beta"},
                {"Date": "2026-08-07", "HomeTeam": " ", "AwayTeam": "Beta"},
            ]
        )

        with self.assertRaisesRegex(MatchDataError, "CSV row 2.*invalid Date"):
            prepare_fixtures(frame)

    def test_duplicate_fixtures_are_rejected_after_trimming_team_names(self) -> None:
        frame = pd.DataFrame(
            [
                {"Date": "07/08/2026", "HomeTeam": "Alpha", "AwayTeam": "Beta"},
                {"Date": "2026-08-07", "HomeTeam": " Alpha ", "AwayTeam": "Beta"},
            ]
        )

        with self.assertRaisesRegex(MatchDataError, "Duplicate fixtures"):
            prepare_fixtures(frame)

    def test_fixture_with_same_home_and_away_team_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [{"Date": "2026-08-07", "HomeTeam": "Alpha", "AwayTeam": "Alpha"}]
        )

        with self.assertRaisesRegex(MatchDataError, "teams must differ"):
            prepare_fixtures(frame)

    def test_batch_prediction_excludes_same_day_and_later_results(self) -> None:
        history = make_history()
        fixture_date = pd.Timestamp("2026-08-07")
        extra_rows = pd.DataFrame(
            [
                {
                    "Date": fixture_date,
                    "HomeTeam": "Alpha",
                    "AwayTeam": "Beta",
                    "FTHG": 100,
                    "FTAG": 0,
                    "FTR": "H",
                },
                {
                    "Date": fixture_date + pd.Timedelta(days=1),
                    "HomeTeam": "Beta",
                    "AwayTeam": "Alpha",
                    "FTHG": 100,
                    "FTAG": 0,
                    "FTR": "H",
                },
            ]
        )
        history = prepare_matches(pd.concat([history, extra_rows], ignore_index=True))
        fixtures = prepare_fixtures(
            pd.DataFrame(
                [{"Date": fixture_date, "HomeTeam": "Alpha", "AwayTeam": "Beta"}]
            )
        )
        prior_results = history[history["Date"] < fixture_date]
        expected = predict_venue_goals("Alpha", "Beta", prior_results)

        predictions = predict_fixtures(fixtures, history)

        self.assertEqual(list(predictions.columns), list(FIXTURE_PREDICTION_COLUMNS))
        self.assertEqual(predictions.iloc[0]["Prediction status"], "Predicted")
        self.assertAlmostEqual(
            predictions.iloc[0]["Expected home goals"],
            expected["expected_home_goals"],
        )
        self.assertAlmostEqual(
            predictions.iloc[0]["Expected away goals"],
            expected["expected_away_goals"],
        )
        self.assertAlmostEqual(
            predictions.iloc[0]["Home-win probability"]
            + predictions.iloc[0]["Draw probability"]
            + predictions.iloc[0]["Away-win probability"],
            1.0,
        )

    def test_insufficient_history_leaves_prediction_fields_blank(self) -> None:
        history = make_history(match_count=4)
        fixtures = prepare_fixtures(
            pd.DataFrame(
                [
                    {
                        "Date": "2026-08-07",
                        "HomeTeam": "Alpha",
                        "AwayTeam": "Beta",
                    }
                ]
            )
        )

        prediction = predict_fixtures(fixtures, history).iloc[0]

        self.assertIn("insufficient earlier history", prediction["Prediction status"])
        self.assertTrue(pd.isna(prediction["Expected home goals"]))
        self.assertTrue(pd.isna(prediction["Home-win probability"]))
        self.assertEqual(prediction["Most likely outcome"], "")

    def test_team_without_prior_results_is_not_predicted(self) -> None:
        history = make_history()
        fixtures = prepare_fixtures(
            pd.DataFrame(
                [
                    {
                        "Date": "2026-08-07",
                        "HomeTeam": "New Team",
                        "AwayTeam": "Alpha",
                    }
                ]
            )
        )

        prediction = predict_fixtures(fixtures, history).iloc[0]

        self.assertIn("no earlier results", prediction["Prediction status"])
        self.assertTrue(pd.isna(prediction["Expected home goals"]))
        self.assertTrue(pd.isna(prediction["Draw probability"]))


if __name__ == "__main__":
    unittest.main()
