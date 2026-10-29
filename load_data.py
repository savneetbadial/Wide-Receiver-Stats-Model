"""
Build a game-by-game wide receiver dataset from nflverse data.

Install:  pip install nflreadpy pandas pyarrow
Run:      python build_wr_dataset.py
Output:   wr_games_raw.csv  (one row = one receiver in one game)
"""

import nflreadpy as nfl
import pandas as pd

SEASONS = list(range(2010, 2026))   # 2010 through the 2025 season
MIN_SEASONS = 3                     # change to 5 for the second iteration
MIN_GAMES_PER_SEASON = 8            # a season only counts if he played this many games

# ---------------------------------------------------------------- load data
# nflreadpy returns polars DataFrames; convert to pandas
stats = nfl.load_player_stats(SEASONS, summary_level="week").to_pandas()
team_stats = nfl.load_team_stats(SEASONS, summary_level="week").to_pandas()
sched = nfl.load_schedules(SEASONS).to_pandas()

# Regular-season WR games only
wr = stats[(stats["position"] == "WR") & (stats["season_type"] == "REG")].copy()
wr = wr[[
    "player_id", "player_display_name", "season", "week", "team", "opponent_team",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_yards_after_catch",
]].rename(columns={
    "player_display_name": "player_name",
    "opponent_team": "opponent",
    "receiving_yards_after_catch": "yac",
})

# Keep players with at least MIN_SEASONS "real" seasons,
# where a real season = at least MIN_GAMES_PER_SEASON games played
games_per_season = wr.groupby(["player_id", "season"]).size()
real_seasons = (games_per_season[games_per_season >= MIN_GAMES_PER_SEASON]
                .groupby("player_id").size())
wr = wr[wr["player_id"].isin(real_seasons[real_seasons >= MIN_SEASONS].index)]

wr = wr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

# ---------------------------------------------------------------- home/away
home = sched[sched["game_type"] == "REG"][["season", "week", "home_team", "away_team"]]
home_map = pd.concat([
    home.rename(columns={"home_team": "team"}).assign(home=1)[["season", "week", "team", "home"]],
    home.rename(columns={"away_team": "team"}).assign(home=0)[["season", "week", "team", "home"]],
])
wr = wr.merge(home_map, on=["season", "week", "team"], how="left")

# --------------------------------------------- receiver rolling form (no leakage)
# shift(1) means each row only sees games BEFORE it.
# Grouping by player only (not player+season) lets windows carry over seasons.
def prior_rolling(df, group, col, n):
    return (df.groupby(group)[col]
              .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

for col, short in [("targets", "targets"), ("receptions", "rec"),
                   ("receiving_yards", "yds"), ("receiving_tds", "td")]:
    wr[f"{short}_avg3"] = prior_rolling(wr, "player_id", col, 3)
    wr[f"{short}_avg8"] = prior_rolling(wr, "player_id", col, 8)
wr["yac_avg8"] = prior_rolling(wr, "player_id", "yac", 8)

wr["games_played_career"] = wr.groupby("player_id").cumcount()  # games before this one

# ---------------------------------------------------------------- team context
ts = team_stats[team_stats["season_type"] == "REG"][["team", "season", "week", "attempts"]]
ts = ts.sort_values(["team", "season", "week"])
ts["team_pass_att_avg3"] = prior_rolling(ts, "team", "attempts", 3)
wr = wr.merge(ts[["team", "season", "week", "team_pass_att_avg3"]],
              on=["team", "season", "week"], how="left")

# ------------------------------------------------------- opponent defense vs WRs
# Total WR receiving yards each defense allowed per game (from all WRs, before filtering
# would be ideal; using the full WR table here)
all_wr = stats[(stats["position"] == "WR") & (stats["season_type"] == "REG")]
d = (all_wr.groupby(["opponent_team", "season", "week"])["receiving_yards"]
           .sum().reset_index()
           .rename(columns={"opponent_team": "opponent", "receiving_yards": "wr_yds_allowed"})
           .sort_values(["opponent", "season", "week"]))
d["opp_wr_yds_allowed_avg5"] = prior_rolling(d, "opponent", "wr_yds_allowed", 5)
# Rank defenses each week: 1 = fewest WR yards allowed (toughest)
d["opp_pass_def_rank"] = (d.groupby(["season", "week"])["opp_wr_yds_allowed_avg5"]
                            .rank(method="min"))
wr = wr.merge(d[["opponent", "season", "week",
                 "opp_wr_yds_allowed_avg5", "opp_pass_def_rank"]],
              on=["opponent", "season", "week"], how="left")

# ---------------------------------------------------------------- final layout
cols = [
    # identifiers
    "player_id", "player_name", "season", "week", "team", "opponent",
    # features
    "home",
    "targets_avg3", "rec_avg3", "yds_avg3", "td_avg3",
    "targets_avg8", "rec_avg8", "yds_avg8", "yac_avg8",
    "team_pass_att_avg3",
    "opp_wr_yds_allowed_avg5", "opp_pass_def_rank",
    "games_played_career",
    # targets
    "receptions", "receiving_yards", "receiving_tds",
]
out = wr[cols].dropna(subset=["yds_avg3"])  # drops each player's very first game
out.to_csv("wr_games_raw.csv", index=False)
print(f"Wrote {len(out):,} rows for {out['player_id'].nunique()} receivers")
