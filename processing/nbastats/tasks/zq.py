import os
import sys
import argparse
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy
import awswrangler as wr
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

sys.path.append("./")

from bla_python_db_utilities.parser import parse_sql
from settings import ENGINE_CONFIG, SQL_PATH, S3_FOLDER
from nbastats.common.encoder import encoder_from_s3
from nbastats.common.playbyplay import (
    column_list,
    encode_player_event,
    collapse_plays,
    preprocess_play,
    add_dob,
    type_dict,
)


def zq_collapse_plays(df, level="possession"):
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
    return result.reset_index(drop=True)


def zq_categorical_encoding(df, from_s3=True):
    """get unique play event"""
    # events = pd.unique(df[column_names("event")].values.ravel())
    # events_series = pd.Series(range(len(events)), events)
    # df[column_names("event")] = df[column_names("event")].apply(
    #     lambda x: events_series[x]
    # )
    player_id_cols = column_list("off_id") + column_list("def_id")
    if from_s3:
        team_encoder = encoder_from_s3("team")
        player_encoder = encoder_from_s3("player")
        # event_encoder = encoder_from_s3("event")
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
    # for event_col in column_list("event"):
    #     df[event_encoder.get_feature_names([event_col])] = event_encoder.transform(
    #         df[[event_col]]
    #     )
    return df


# def add_dob(df, player):
#     """summarize data frame"""
#     df["Pts"] = np.where(df["HomeOff"] == 1, df["HomePts"], df["AwayPts"])
#     player_abbr = column_list("off_id_abbr") + column_list("def_id_abbr")
#     df = df.rename(
#         columns=dict(zip(column_list("off_id") + column_list("def_id"), player_abbr,))
#     )
#     # df["TimeRemain"] = df["SecRemainGame"]

#     # df.loc[df["Pts"] > 3, "Pts"] = 3
#     birth_dates = player[["PlayerId", "Dob"]]
#     for nth in range(1, 11):
#         df = pd.merge(
#             df,
#             birth_dates.rename(columns={"PlayerId": f"P{nth}", "Dob": f"Age{nth}"}),
#             on=f"P{nth}",
#             how="left",
#         )

#     df[column_list("age")] = df[column_list("age")].apply(
#         lambda s: (df["GameDate"] - s).dt.days / 365.25
#     )
#     return df


def zq_output(df):
    """summarise the data"""
    return df.astype(type_dict)


def zq_pipeline(level="possession"):
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    # team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    # game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    player = pd.read_sql(parse_sql(SQL_PATH["player"], False), engine)
    play = preprocess_play(play)
    play = encode_player_event(play, from_s3=True)
    collapsed = collapse_plays(play, level=level, event=True)
    result = add_dob(collapsed, player)
    result = zq_categorical_encoding(result, from_s3=True)
    return zq_output(result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--level", type=str, default="possession")
    args.add_argument("--output_mode", type=str, default="")
    args = args.parse_args()
    df = zq_pipeline(level=args.level)
    if args.output_mode:
        wr.s3.to_parquet(
            df=df,
            path=S3_FOLDER + f"zq_{args.level}",
            dataset=True,
            mode=args.output_mode,
            # table="proc_play",
            # database="nbastats",
        )
        print("Uploaded parquet file to S3.")
    else:
        print(df.head())
        print(df.dtypes)
