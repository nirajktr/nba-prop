# NBA Prop Betting Analytics

This project analyzes game-level betting market data (MGM lines via Yahoo internal API) to build predictive models for NBA betting markets.

## Scope

- Markets: full-game over/under, moneyline, and spread
- Coverage: 2021-22 through 2024-25 full seasons, plus partial 2025-26 through 2026-02-12
- Current baseline backtest window: 2021-22 through 2024-25

## Data Notes

- Each row is one NBA game.
- Odds in this dataset are closing-line odds.
- Games missing Yahoo betting data were omitted.
- The file includes both wager percentage (ticket share) and stake percentage (money share).

## Data Dictionary

- `game_id`: Yahoo internal game ID used in source API calls.
- `game_date`, `away_team`, `home_team`: game metadata.
- `pregame_odds`: Yahoo display summary of spread/total pregame line.

Totals (over/under):

- `total_over_points`: full-game over/under points line.
- `total_over_stake_percentage`: percent of total money risked on over.
- `total_over_wager_percentage`: percent of total bets on over.
- `total_over_odds`: over odds (American format).
- `total_over_decimal_odds`: over odds (decimal format).
- `total_over_won`: whether over won.
- `total_under_*`: same fields for under side.

Moneyline:

- `money_away_odds`: away moneyline odds (American format).
- `money_away_decimal_odds`: away moneyline odds (decimal format).
- `money_away_stake_percentage`: percent of money on away moneyline.
- `money_away_wager_percentage`: percent of tickets on away moneyline.
- `money_away_won`: whether away moneyline won.
- `money_home_*`: same fields for home side.

Spread:

- `spread_away_points`: away team spread handicap (for example, `-6` means away must win by 7+).
- `spread_away_odds`: away spread odds (American format).
- `spread_away_decimal_odds`: away spread odds (decimal format).
- `spread_away_stake_percentage`: percent of money on away against the spread.
- `spread_away_wager_percentage`: percent of tickets on away against the spread.
- `spread_away_won`: whether away side covered.
- `spread_home_*`: same fields for home side.

## NBA API Smoke Test

Run:

```bash
uv run python src/main.py
```

This verifies `nba_api` import and a basic data fetch from static team metadata.

## Baseline Backtest (Logistic Regression)

Run:

```bash
uv run python src/backtest.py
```

What it does:

- Filters seasons to 2021-22 through 2024-25.
- Uses rolling season backtests: train on prior seasons, test on next season.
- Trains logistic regression models for:
	- over winner (`total_over_won`)
	- home moneyline winner (`money_home_won`)
	- home spread winner (`spread_home_won`)
- Reports classification metrics and a simple edge-based betting simulation.

Current betting simulation rule:

- Place a bet when model probability exceeds implied probability by at least `0.03`.
- Stake is 1 unit per qualified bet.

## Next Modeling Steps

- Add calibration (Platt or isotonic) before converting probabilities to bet decisions.
- Introduce time-aware feature lags and rolling team performance features.
- Compare against gradient-boosted trees and market-implied baselines.
- Tune edge thresholds per market on validation folds.