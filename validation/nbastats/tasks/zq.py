import os
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy

from bla_python_db_utilities.parser import parse_sql
from settings import ENGINE_CONFIG, SQL_PATH
from nbastats.common.playbyplay import column_names, convert_homeaway_to_offdef


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


def summarize_data(df):
    """summarize data frame"""
    pass

def processing():
    """data processing"""
    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    return convert_homeaway_to_offdef(play)


if __name__ == "__main__":
    processing()
