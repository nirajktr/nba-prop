from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "all_odds.csv"
SEASON_START_MIN = 2021
SEASON_START_MAX = 2024
EDGE_THRESHOLD = 0.02


@dataclass
class MarketSpec:
    name: str
    target_col: str
    decimal_odds_col: str


MARKETS: tuple[MarketSpec, ...] = (
    MarketSpec("total_over", "total_over_won", "total_over_decimal_odds"),
    MarketSpec("money_home", "money_home_won", "money_home_decimal_odds"),
    MarketSpec("spread_home", "spread_home_won", "spread_home_decimal_odds"),
)


FEATURE_COLS = [
    "total_over_points",
    "total_over_stake_percentage",
    "total_over_wager_percentage",
    "total_over_odds",
    "total_over_decimal_odds",
    "total_under_points",
    "total_under_stake_percentage",
    "total_under_wager_percentage",
    "total_under_odds",
    "total_under_decimal_odds",
    "money_away_odds",
    "money_away_decimal_odds",
    "money_away_stake_percentage",
    "money_away_wager_percentage",
    "money_home_odds",
    "money_home_decimal_odds",
    "money_home_stake_percentage",
    "money_home_wager_percentage",
    "spread_away_points",
    "spread_away_odds",
    "spread_away_decimal_odds",
    "spread_away_stake_percentage",
    "spread_away_wager_percentage",
    "spread_home_points",
    "spread_home_odds",
    "spread_home_decimal_odds",
    "spread_home_stake_percentage",
    "spread_home_wager_percentage",
]


def compute_season_start(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, format="%Y-%m-%d-%H:%M", errors="coerce")
    return dt.dt.year.where(dt.dt.month >= 7, dt.dt.year - 1)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove the CSV index column that is persisted in the file.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["season_start_year"] = compute_season_start(df["game_date"])
    df = df[df["season_start_year"].between(SEASON_START_MIN, SEASON_START_MAX)]
    return df


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(log_loss(y_true, y_prob))


def implied_probability(decimal_odds: pd.Series) -> pd.Series:
    return 1.0 / decimal_odds


def calculate_units(y_true: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    win_units = decimal_odds - 1.0
    lose_units = -1.0
    return y_true.astype(float) * win_units + (1.0 - y_true.astype(float)) * lose_units


def evaluate_market(df: pd.DataFrame, market: MarketSpec) -> pd.DataFrame:
    rows: list[dict] = []

    for test_season in range(SEASON_START_MIN + 1, SEASON_START_MAX + 1):
        train_df = df[df["season_start_year"] < test_season].copy()
        test_df = df[df["season_start_year"] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        train_df = train_df.dropna(subset=[market.target_col])
        test_df = test_df.dropna(subset=[market.target_col, market.decimal_odds_col])

        if train_df.empty or test_df.empty:
            continue

        x_train = train_df[FEATURE_COLS]
        y_train = train_df[market.target_col].astype(int)
        x_test = test_df[FEATURE_COLS]
        y_test = test_df[market.target_col].astype(int)

        model = build_model()
        model.fit(x_train, y_train)

        y_prob = pd.Series(model.predict_proba(x_test)[:, 1], index=test_df.index)
        y_pred = (y_prob >= 0.5).astype(int)

        # Bet only when model probability exceeds implied probability by edge threshold.
        implied = implied_probability(test_df[market.decimal_odds_col])
        bet_mask = (y_prob - implied) >= EDGE_THRESHOLD

        units = calculate_units(y_test, test_df[market.decimal_odds_col])
        units_bet = units[bet_mask]

        rows.append(
            {
                "market": market.name,
                "test_season": f"{test_season}-{test_season + 1}",
                "n_games": int(len(test_df)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "log_loss": safe_log_loss(y_test, y_prob),
                "roc_auc": safe_roc_auc(y_test, y_prob),
                "n_bets": int(bet_mask.sum()),
                "total_units": float(units_bet.sum()) if len(units_bet) else 0.0,
                "roi_per_bet": float(units_bet.mean()) if len(units_bet) else 0.0,
                "edge_threshold": EDGE_THRESHOLD,
            }
        )

    return pd.DataFrame(rows)


def summarize(results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    all_results = pd.concat(list(results), ignore_index=True)
    return all_results.sort_values(["market", "test_season"]).reset_index(drop=True)


def main() -> None:
    df = load_data(DATA_PATH)

    if df.empty:
        raise SystemExit("No rows found after season filtering. Check game_date formatting.")

    results = [evaluate_market(df, market) for market in MARKETS]
    summary_df = summarize(results)

    print("=== Backtest Results (Logistic Regression Baseline) ===")
    print(summary_df.to_string(index=False))

    agg = (
        summary_df.groupby("market", as_index=False)
        .agg(
            n_games=("n_games", "sum"),
            n_bets=("n_bets", "sum"),
            total_units=("total_units", "sum"),
            avg_accuracy=("accuracy", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_roi_per_bet=("roi_per_bet", "mean"),
        )
        .sort_values("market")
    )

    print("\n=== Aggregate by Market ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
