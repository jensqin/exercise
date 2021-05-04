import os
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy

from bla_python_db_utilities.parser import parse_sql

from nbastats.schema.playbyplay import play_schema, agg_schema
from settings import ENGINE_CONFIG, SQL_PATH

# engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])

# game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
# play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)

# sc = pa.infer_schema(player)
# sc = pa.infer_schema(team)
# print(sc.to_script())

# play by play data types

# GameId                  int64
# PlayNum                 int64
# Eventnum                int64
# Eventmsgtype            int64
# Season                  int64
# HomeTeamId            float64
# HomePlayer1Id         float64
# HomePlayer2Id         float64
# HomePlayer3Id         float64
# HomePlayer4Id         float64
# HomePlayer5Id         float64
# AwayTeamId            float64
# AwayPlayer1Id         float64
# AwayPlayer2Id         float64
# AwayPlayer3Id         float64
# AwayPlayer4Id         float64
# AwayPlayer5Id         float64
# SecRemainGame           int64
# Period                  int64
# SecRemainPeriod         int64
# ScoreMargin             int64
# HomeOff               float64
# StartEvent             object
# EndEvent               object
# HomePlayer1Event       object
# HomePlayer2Event       object
# HomePlayer3Event       object
# HomePlayer4Event       object
# HomePlayer5Event       object
# AwayPlayer1Event       object
# AwayPlayer2Event       object
# AwayPlayer3Event       object
# AwayPlayer4Event       object
# AwayPlayer5Event       object
# HomeNumPlayers        float64
# AwayNumPlayers        float64
# HomeFouls               int64
# AwayFouls               int64
# HomeScore               int64
# AwayScore               int64
# PossCount             float64
# SecSinceLastPlay        int64
# Eventmsgactiontype      int64
# dtype: object


def column_names(key):
    """get column names"""
    name_dict = {
        "id": [
            "HomePlayer1Id",
            "HomePlayer2Id",
            "HomePlayer3Id",
            "HomePlayer4Id",
            "HomePlayer5Id",
            "AwayPlayer1Id",
            "AwayPlayer2Id",
            "AwayPlayer3Id",
            "AwayPlayer4Id",
            "AwayPlayer5Id",
        ],
        "event": [
            "HomePlayer1Event",
            "HomePlayer2Event",
            "HomePlayer3Event",
            "HomePlayer4Event",
            "HomePlayer5Event",
            "AwayPlayer1Event",
            "AwayPlayer2Event",
            "AwayPlayer3Event",
            "AwayPlayer4Event",
            "AwayPlayer5Event",
        ],
        "home_event": [
            "HomePlayer1Event",
            "HomePlayer2Event",
            "HomePlayer3Event",
            "HomePlayer4Event",
            "HomePlayer5Event",
        ],
        "away_event": [
            "AwayPlayer1Event",
            "AwayPlayer2Event",
            "AwayPlayer3Event",
            "AwayPlayer4Event",
            "AwayPlayer5Event",
        ],
        "home_id": [
            "HomePlayer1Id",
            "HomePlayer2Id",
            "HomePlayer3Id",
            "HomePlayer4Id",
            "HomePlayer5Id",
        ],
        "away_id": [
            "AwayPlayer1Id",
            "AwayPlayer2Id",
            "AwayPlayer3Id",
            "AwayPlayer4Id",
            "AwayPlayer5Id",
        ],
        "off_id": [
            "OffPlayer1Id",
            "OffPlayer2Id",
            "OffPlayer3Id",
            "OffPlayer4Id",
            "OffPlayer5Id",
        ],
        "def_id": [
            "DefPlayer1Id",
            "DefPlayer2Id",
            "DefPlayer3Id",
            "DefPlayer4Id",
            "DefPlayer5Id",
        ],
        "off_id_abbr": ["P1", "P2", "P3", "P4", "P5"],
        "def_id_abbr": ["P6", "P7", "P8", "P9", "P10"],
        "off_age": ["Age1", "Age2", "Age3", "Age4", "Age5"],
        "def_age": ["Age6", "Age7", "Age8", "Age7", "Age10"],
        "age": [
            "Age1",
            "Age2",
            "Age3",
            "Age4",
            "Age5",
            "Age6",
            "Age7",
            "Age8",
            "Age7",
            "Age10",
        ],
    }
    return name_dict[key]


@pa.check_io(output=play_schema)
def preprocess_play(df):
    """preprocess play"""

    # important
    df = df.sort_values(["GameId", "PlayNum"], ascending=True).reset_index(drop=True)

    # home/away to off/def
    df["OffensiveTeamId"] = np.select(
        [df["HomeOff"].isna(), df["HomeOff"] == 1, df["HomeOff"] == 0],
        [None, df["HomeTeamId"], df["AwayTeamId"]],
    )
    df["DefensiveTeamId"] = np.select(
        [df["HomeOff"].isna(), df["HomeOff"] == 1, df["HomeOff"] == 0],
        [None, df["AwayTeamId"], df["HomeTeamId"]],
    )

    # HomeScore/AwayScore not na
    df[["HomePts", "AwayPts"]] = (
        df.groupby("GameId")[["HomeScore", "AwayScore"]].diff(periods=1).fillna(0)
    )

    # TODO: detect errors, pandera
    # df.loc[(df["HomePts"] < 0) | (df["AwayPts"] < 0), ["HomePts", "AwayPts"]] = 0
    assert (df["HomePts"] > 0).all()
    assert (df["AwayPts"] > 0).all()
    assert (df["ScoreMargin"] == df["HomeScore"] - df["AwayScore"]).all()
    assert df["PossCount"].notna().all()

    # df = df.loc[df["EndEvent"] != "EndPeriod"]

    # # remove plays without off team
    # df = df.loc[df["HomeOff"].isna() | df["EndEvent"] != "Timeout"]

    # df = df.loc[(df["HomeNumPlayers"] == 5) & (df["AwayNumPlayers"] == 5)]

    return df


@pa.check_output(agg_schema)
def aggregation_to_game_level(play, game):
    """clean by summation"""
    # play = play.loc[~((play["HomePts"] > 0) & (play["HomeOff"] == 1))]
    # play = play.loc[~((play["AwayPts"] > 0) & (play["HomeOff"] == 0))]
    final_scores = (
        play.groupby("GameId")[["HomeScore", "AwayScore"]]
        .agg(["max", "last"])
        .reset_index()
    )

    # score of last play equals max score
    # assert (
    #     final_scores[("HomeScore", "max")] == final_scores[("HomeScore", "last")]
    # ).all()
    # assert (
    #     final_scores[("HomeScore", "max")] == final_scores[("HomeScore", "last")]
    # ).all()

    final_scores.columns = [
        "_".join(col).rstrip("_") for col in final_scores.columns.values
    ]
    return pd.merge(
        final_scores, game[["GameId", "HomeFinalScore", "AwayFinalScore"]], on="GameId",
    )


def filter_regular_plays(df):
    """filter out irregular plays"""

    # potential issue: time calculation
    # homeoff is null: mostly timeout or ejection
    df = df.loc[(df["EndEvent"] != "EndPeriod") & df["HomeOff"].notna()]

    # remove plays without off team
    # df = df.loc[df["HomeOff"].isna() | df["EndEvent"] != "Timeout"]

    # missing players: 4.5%
    # optional for zq
    bad_possession = df.loc[
        (df["HomeNumPlayers"] == 5) & (df["AwayNumPlayers"] == 5),
        ["GameId", "PossCount"],
    ]
    df = df.loc[~df[["GameId", "PossCount"]].isin(bad_possession)]

    # df.loc[df["SecSinceLastPlay"] > 24, "SecSinceLastPlay"] = 24
    df.loc[df["SecSinceLastPlay"] > 30, "SecSinceLastPlay"] = 30
    df.loc[df["SecSinceLastPlay"] < 0, "SecSinceLastPlay"] = 0

    return df


def players_of_possession(df):
    """players who use most of the possession"""
    df["SumSec"] = df.groupby(["GameId", "PossCount"] + column_names("id"))[
        "SecSinceLastPlay"
    ].transform(sum)
    df_id = (
        df.sort_values(["GameId", "PossCount", "SumSec"], ascending=[True, True, False])
        .groupby(["GameId", "PossCount"])[column_names("id")]
        .first()
    )
    return pd.merge(
        df.drop(columns=column_names("id")), df_id, on=["GameId", "PossCount"]
    )


def encode_play_event(df):
    """get unique play event"""
    events = pd.unique(df[column_names("event")].values.ravel())
    events_series = pd.Series(range(len(events)), events)
    df[column_names("event")] = df[column_names("event")].apply(
        lambda x: events_series[x]
    )
    return df


def collapse_plays(df, level="possession"):
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
            "Eventnum",
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
            Eventnum_min=("Eventnum", "min"),
            Eventnum_max=("Eventnum", "max"),
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
    # result["ScoreMargin"] = (
    #     result["HomeScore"]
    #     - result["AwayScore"]
    #     - result["HomePts"]
    #     + result["AwayPts"]
    # )
    result["ScoreMargin"] = result.groupby("GameId")["ScoreMargin"].shift(1).fillna(0)
    result["ScoreMargin"] = result["ScoreMargin"] * np.where(
        result["HomeOff"] == 1, 1, -1
    )
    return result.reset_index()


def processing():
    """data processing"""
    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    return preprocess_play(play)


if __name__ == "__main__":
    """main script"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    processing()
