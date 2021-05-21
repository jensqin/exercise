import os
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
import sqlalchemy

from bla_python_db_utilities.parser import parse_sql

from nbastats.schema.playbyplay import play_schema
from settings import ENGINE_CONFIG, SQL_PATH


def column_list(key):
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
        "def_age": ["Age6", "Age7", "Age8", "Age9", "Age10"],
        "age": [
            "Age1",
            "Age2",
            "Age3",
            "Age4",
            "Age5",
            "Age6",
            "Age7",
            "Age8",
            "Age9",
            "Age10",
        ],
    }
    return name_dict[key]


@pa.check_input(play_schema)
def preprocess_play(df):
    """preprocess play"""

    # important
    # df = df.sort_values(["GameId", "PlayNum"], ascending=True).reset_index(drop=True)
    df = df.reset_index(drop=True)

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
    df.loc[(df["HomePts"] < 0) | (df["AwayPts"] < 0), ["HomePts", "AwayPts"]] = 0
    assert (df["HomePts"] >= 0).all()
    assert (df["AwayPts"] >= 0).all()

    # TODO: solve data issue
    df["ScoreMargin"] = df["HomeScore"] - df["AwayScore"]
    # assert (df["ScoreMargin"] == df["HomeScore"] - df["AwayScore"]).all()

    df = df.loc[df["EndEvent"] != "EndPeriod"]
    # assert df["PossCount"].notna().all()

    # # remove plays without off team
    df = df.loc[df["HomeOff"].isna() | df["EndEvent"] != "Timeout"]

    # df = df.loc[(df["HomeNumPlayers"] == 5) & (df["AwayNumPlayers"] == 5)]

    # potential issue: time calculation
    # homeoff is null: mostly timeout or ejection
    df = df.loc[(df["EndEvent"] != "EndPeriod") & df["HomeOff"].notna()]

    # remove plays without off team
    # df = df.loc[df["HomeOff"].isna() | df["EndEvent"] != "Timeout"]

    # missing players: 4.5%
    # optional for zq
    bad_possession = df.loc[
        (df["HomeNumPlayers"] != 5) | (df["AwayNumPlayers"] != 5),
        ["GameId", "PossCount"],
    ].drop_duplicates()
    # df = df.loc[~df[["GameId", "PossCount"]].isin(bad_possession).all()]
    df = pd.merge(
        df, bad_possession, how="left", on=["GameId", "PossCount"], indicator=True
    )
    df = df.loc[df["_merge"] == "left_only"].drop(columns="_merge")

    # df.loc[df["SecSinceLastPlay"] > 24, "SecSinceLastPlay"] = 24
    df.loc[df["SecSinceLastPlay"] > 30, "SecSinceLastPlay"] = 30
    df.loc[df["SecSinceLastPlay"] < 0, "SecSinceLastPlay"] = 0

    return df


# @pa.check_output(agg_schema)
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


# def filter_regular_plays(df):
#     """filter out irregular plays"""

#     # potential issue: time calculation
#     # homeoff is null: mostly timeout or ejection
#     df = df.loc[(df["EndEvent"] != "EndPeriod") & df["HomeOff"].notna()]

#     # remove plays without off team
#     # df = df.loc[df["HomeOff"].isna() | df["EndEvent"] != "Timeout"]

#     # missing players: 4.5%
#     # optional for zq
#     bad_possession = df.loc[
#         (df["HomeNumPlayers"] != 5) | (df["AwayNumPlayers"] != 5),
#         ["GameId", "PossCount"],
#     ].drop_duplicates()
#     # df = df.loc[~df[["GameId", "PossCount"]].isin(bad_possession).all()]
#     df = pd.merge(
#         df, bad_possession, how="left", on=["GameId", "PossCount"], indicator=True
#     )
#     df = df.loc[df["_merge"] == "left_only"].drop(columns="_merge")

#     # df.loc[df["SecSinceLastPlay"] > 24, "SecSinceLastPlay"] = 24
#     df.loc[df["SecSinceLastPlay"] > 30, "SecSinceLastPlay"] = 30
#     df.loc[df["SecSinceLastPlay"] < 0, "SecSinceLastPlay"] = 0

#     return df


def players_of_possession(df):
    """players who use most of the possession"""
    df["SumSec"] = df.groupby(["GameId", "PossCount"] + column_list("id"))[
        "SecSinceLastPlay"
    ].transform(sum)
    df_id = (
        df.sort_values(["GameId", "PossCount", "SumSec"], ascending=[True, True, False])
        .groupby(["GameId", "PossCount"])[column_list("id")]
        .first()
    )
    return pd.merge(
        df.drop(columns=column_list("id")), df_id, on=["GameId", "PossCount"]
    )


def encode_play_event(df):
    """get unique play event"""
    events = pd.unique(df[column_list("event")].values.ravel())
    events_series = pd.Series(range(len(events)), events)
    df[column_list("event")] = df[column_list("event")].apply(
        lambda x: events_series[x]
    )
    return df


def collapse_plays(df, level="possession"):
    """collapse play level"""
    # check homeoff consistency
    collapsed = df.loc[
        :,
        [
            "HomePts",
            "AwayPts",
            "HomeFouls",
            "AwayFouls",
            # "ScoreMargin",
            "SecRemainGame",
            "SecSinceLastPlay",
            "GameId",
            "HomeOff",
            "Period",
            "Season",
            "PossCount",
            "StartEvent",
            "EndEvent",
            "GameDate",
            "GameType",
            "ShotDistance",
            "ShotAngle",
            "OffensiveTeamId",
            "DefensiveTeamId",
            "PlayNum",
            "HomeScore",
            "AwayScore",
        ],
    ]
    if level == "chance":
        collapsed["ChanceCount"] = np.where(collapsed["StartEvent"] == "Oreb", 1, 0)
        collapsed["PossCount"] = (
            collapsed.groupby("GameId")["ChanceCount"].cumsum() + collapsed["PossCount"]
        )
        # collapsed = collapsed.drop(columns="PossCount").rename(
        #     columns={"ChanceCount": "PossCount"}
        # )
    elif level != "possession":
        raise ValueError(f"level must be possession or chance, but get {level}.")
    result = collapsed.groupby(["GameId", "PossCount"]).agg(
        Season=("Season", "first"),
        GameDate=("GameDate", "first"),
        GameType=("GameType", "first"),
        OffTeam=("OffensiveTeamId", "first"),
        DefTeam=("DefensiveTeamId", "first"),
        # ScoreMargin=("ScoreMargin", "first"),
        Period=("Period", "first"),
        HomeOff=("HomeOff", "first"),
        SecRemainGame=("SecRemainGame", "max"),
        StartPlayNum=("PlayNum", "min"),
        StartEvent=("StartEvent", "first"),
        EndEvent=("EndEvent", "last"),
        HomeScore=("HomeScore", "max"),
        AwayScore=("AwayScore", "max"),
        HomePts=("HomePts", "sum"),
        AwayPts=("AwayPts", "sum"),
        Duration=("SecSinceLastPlay", "sum"),
        ShotDistance=("ShotDistance", "mean"),
        ShotAngle=("ShotAngle", "mean"),
    )
    # TODO: use original scoremargin
    result["ScoreMargin"] = (
        result["HomeScore"]
        - result["AwayScore"]
        - result["HomePts"]
        + result["AwayPts"]
    )
    # result["ScoreMargin"] = result.groupby("GameId")["ScoreMargin"].shift(1).fillna(0)
    result["ScoreMargin"] = result["ScoreMargin"] * np.where(
        result["HomeOff"] == 1, 1, -1
    )
    return result.reset_index()


# def load_data(name):
#     """data processing"""
#     engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
#     return pd.read_sql(parse_sql(SQL_PATH[name], False), engine)


def common_play(play, level="possession"):
    """ common processing for play"""
    # play = load_data("play")
    df = preprocess_play(play)
    # df = filter_regular_plays(df)
    return df, collapse_plays(df, level=level)


if __name__ == "__main__":
    """main script"""
    print("This is common script.")
