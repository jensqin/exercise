import os
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from bla_python_db_utilities.parser import parse_sql
from settings import ENGINE_CONFIG, SQL_PATH
from nbastats.common.encoder import encoder_from_s3
from nbastats.common.playbyplay import column_names, common_play


def collapse_chance_level(df, level="chance"):
    """collapse play level"""
    # check homeoff consistency
    collapsed = df[
        [
            "ScoreMargin",
            "HomePts",
            "AwayPts",
            "HomeFouls",
            "AwayFouls",
            "SecRemainGame",
            "SecSinceLastPlay",
            "GameId",
            "HomeOff",
            "Period",
            "Season",
            "PossCount",
            "OffensiveTeamId",
            "DefensiveTeamId",
            "PlayNum",
            "HomeScore",
            "AwayScore",
        ]
    ]
    if level == "possession":
        result = collapsed.groupby(["GameId", "PossCount"]).agg(
            Season=("Season", "first"),
            Period=("Period", "first"),
            HomeOff=("HomeOff", "first"),
            SecRemainGame=("SecRemainGame", "max"),
            PlayNum_min=("PlayNum", "min"),
            PlayNum_max=("PlayNum", "max"),
            HomeScore=("HomeScore", "max"),
            AwayScore=("AwayScore", "max"),
            HomePts=("HomePts", "sum"),
            AwayPts=("AwayPts", "sum"),
            SecSinceLastPlay=("SecSinceLastPlay", "sum"),
        )
    elif level == "chance":
        collapsed["ChanceCount"] = np.where(collapsed["StarEvent"] == "Oreb", 1, 0)
        collapsed["ChanceCount"] = (
            collapsed.groupby("GameId")["ChanceCount"].cumsum() + collapsed["PossCount"]
        )
        result = collapsed.groupby(["GameId", "ChanceCount"]).agg(
            Season=("Season", "first"),
            Period=("Period", "first"),
            HomeOff=("HomeOff", "first"),
            SecRemainGame=("SecRemainGame", "max"),
            PlayNum_min=("PlayNum", "min"),
            PlayNum_max=("PlayNum", "max"),
            Eventnum_min=("Eventnum", "min"),
            Eventnum_max=("Eventnum", "max"),
            HomeScore=("HomeScore", "max"),
            AwayScore=("AwayScore", "max"),
            HomePts=("HomePts", "sum"),
            AwayPts=("AwayPts", "sum"),
            SecSinceLastPlay=("SecSinceLastPlay", "sum"),
        )
    else:
        raise ValueError(f"level must be possession or chance, but get {level}.")

    # TODO: use original scoremargin
    result["ScoreMargin"] = (
        result["HomeScore"]
        - result["AwayScore"]
        - result["HomePts"]
        + result["AwayPts"]
    )
    result["ScoreMargin"] = result["ScoreMargin"] * np.where(
        result["HomeOff"] == 1, 1, -1
    )
    return result.reset_index()


def categorical_encoding(df, from_s3=True):
    """get unique play event"""
    # events = pd.unique(df[column_names("event")].values.ravel())
    # events_series = pd.Series(range(len(events)), events)
    # df[column_names("event")] = df[column_names("event")].apply(
    #     lambda x: events_series[x]
    # )
    player_id_cols = (
        column_names("off_id_abbr") + column_names("def_id_abbr")
    )
    if from_s3:
        team_encoder = encoder_from_s3("team")
        player_encoder = encoder_from_s3("player")
        event_encoder = encoder_from_s3("event")
    else:
        team_encoder = OrdinalEncoder()
        team_encoder.fit(df[["OffTeam", "DefTeam"]])
        player_encoder = OrdinalEncoder()
        player_encoder.fit(df[player_id_cols])
    # df[["OffTeam", "DefTeam"]] = df[["OffTeam", "DefTeam"]].apply(
    #     lambda t: team_encoder.transform(t.to_frame())
    # )
    for team_col in ["OffTeam", "DefTeam"]:
        df[team_col] = team_encoder.transform(df[[team_col]])
    for player_col in player_id_cols:
        df[player_col] = player_encoder.transform(df[[player_col]])
    for event_col in column_names("event"):
        df[event_encoder.get_feature_names([event_col])] = event_encoder.transform(
            df[[event_col]]
        )
    # df["OffTeam"] = team_encoder.transform(df[["OffTeam"]])
    # df[player_id_cols] = player_encoder.transform(df[player_id_cols])
    return df


def summarise_data(df, player):
    """summarize data frame"""
    df["Pts"] = np.where(df["HomeOff"] == 1, df["HomePts"], df["AwayPts"])
    df = df.rename(
        columns=dict(
            zip(
                column_names("off_id") + column_names("def_id"),
                column_names("off_id_abbr") + column_names("def_id_abbr"),
            )
        )
    )
    # df["TimeRemain"] = df["SecRemainGame"]

    # df.loc[df["Pts"] > 3, "Pts"] = 3
    birth_dates = player[["PlayerId", "Dob"]]
    for nth in range(1, 11):
        df = pd.merge(
            df,
            birth_dates.rename(columns={"PlayerId": f"P{nth}", "Dob": f"Age{nth}"}),
            on=f"P{nth}",
            how="left",
        )

    df[column_names("age")] = df[column_names("age")].apply(
        lambda s: (df["GameDate"] - s).dt.days / 365.25
    )
    return df


def zq_pipeline():
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    # team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    # game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    player = pd.read_sql(parse_sql(SQL_PATH["player"], False), engine)
    play, possession = common_play(play)
    result = summarise_data(possession, player)
    return categorical_encoding(result, from_s3=True)


if __name__ == "__main__":
    zq_pipeline()
