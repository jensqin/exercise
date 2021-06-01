import sys
import os
import argparse
from bla_python_db_utilities import parser
import numpy as np
import pandas as pd
import awswrangler as wr
import pandera as pa
import sqlalchemy
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from bla_python_db_utilities.parser import parse_sql

sys.path.append("./")

from settings import ENGINE_CONFIG, SQL_PATH, S3_FOLDER
from nbastats.common.encoder import encoder_from_s3
from nbastats.common.playbyplay import (
    column_list,
    preprocess_play,
    collapse_plays,
    add_dob,
    type_dict,
)


def usage_player_id(df, collapsed):
    """get usage player id"""

    # remove ast as event
    # event_columns = [
    #     f"{x}Player{y}Event" for x in ["Home", "Away"] for y in range(1, 6)
    # ]
    df = df.loc[
        :,
        ["GameId", "PossCount", "HomeOff", "Eventmsgtype"]
        + column_list("offdef_id")
        + column_list("offdef_event"),
    ]
    df.loc[df["Eventmsgtype"].isin([1, 2, 3, 5]), column_list("offdef_event")] = df.loc[
        df["Eventmsgtype"].isin([1, 2, 3, 5]), column_list("offdef_event")
    ].replace({"Ast": None})
    df = df.drop(columns="Eventmsgtype")

    # # insert 0 for null playerId
    # id_columns = [f"{x}Player{y}Id" for x in ["Home", "Away"] for y in range(1, 6)]
    # df[column_names("id")] = df[column_names("id")].fillna(0)

    # only one player uses a possession
    assert (df[column_list("off_event")].notna().sum(axis=1) < 2).all()

    df["UsageId"] = np.where(
        df[column_list("off_event")].notna(), df[column_list("off_id")], 0,
    ).sum(axis=1)

    # df["PlayerId"] = df["PlayerId"].fillna(0)
    # df["UsageId"] = np.where(df["HomeOff"] == 1, UsgId1, UsgId0)

    df["UsageId"] = (
        df.loc[df["UsageId"] > 0]
        .groupby(["GameId", "PossCount"])["UsageId"]
        .transform("last")
    )

    # homeaway to offdef
    # df.loc[df["HomeOff"] == 1, column_list("off_id") + column_list("def_id")] = df.loc[
    #     df["HomeOff"] == 1, column_list("home_id") + column_list("away_id")
    # ].values
    # df.loc[df["HomeOff"] == 0, column_list("off_id") + column_list("def_id")] = df.loc[
    #     df["HomeOff"] == 0, column_list("away_id") + column_list("home_id")
    # ].values
    # df[column_list("off_id") + column_list("def_id")] = df[
    #     column_list("off_id") + column_list("def_id")
    # ].astype("int")
    # df[column_names("off_id") + column_names("def_id")] = np.where(
    #     df["HomeOff"] == 1,
    #     df[column_names("home_id") + column_names("away_id")].values,
    #     df[column_names("away_id") + column_names("home_id")].values,
    # )
    df = df.loc[
        df[column_list("off_id")]
        .apply(lambda t: (df["UsageId"] == t) | df["UsageId"].isna())
        .any(axis=1)
    ]
    poss_usg = df.loc[
        df["UsageId"].notna(), ["GameId", "PossCount", "UsageId"]
    ].drop_duplicates()
    # assert len(poss_usg.index) == len(
    #     poss_usg[["GameId", "PossCount"]].drop_duplicates().index
    # )

    # result = pd.merge(
    #     collapsed[
    #         [
    #             "GameId",
    #             "PossCount",
    #             "GameDate",
    #             "GameType",
    #             "OffTeam",
    #             "DefTeam",
    #             "HomeOff",
    #             "ScoreMargin",
    #             "Duration",
    #             "HomePts",
    #             "AwayPts",
    #             "ShotDistance",
    #             "ShotAngle",
    #         ]
    #     ],
    #     df[
    #         ["GameId", "PossCount"] + column_list("off_id") + column_list("def_id")
    #     ].drop_duplicates(),
    #     on=["GameId", "PossCount"],
    #     how="left",
    # )
    result = pd.merge(collapsed, poss_usg, on=["GameId", "PossCount"], how="left")
    assert len(result.index) == len(
        collapsed.index
    ), "Number of rows changed after merging!"

    # TODO: apply PFoul split to substitutes in possessions
    # 0.9957044673539519
    # print(
    #     result[column_list("off_id")]
    #     .apply(lambda t: (result["UsageId"] == t) | result["UsageId"].isna())
    #     .any(axis=1)
    #     .mean()
    # )
    result = result[
        result[column_list("off_id")]
        .apply(lambda t: (result["UsageId"] == t) | result["UsageId"].isna())
        .any(axis=1)
    ]
    assert (
        result[column_list("off_id")]
        .apply(lambda t: (result["UsageId"] == t) | result["UsageId"].isna())
        .any(axis=1)
        .all()
    ), "UsageId not in offensive Ids."
    return result


def zb_categorical_encoding(df, from_s3=False):
    """encode categorical variables"""
    player_id_cols = column_list("off_id") + column_list("def_id")
    if from_s3:
        team_encoder = encoder_from_s3("team")
        player_encoder = encoder_from_s3("player")
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
    df["UsageId"] = player_encoder.transform(df[["UsageId"]].fillna(-1))
    # df["OffTeam"] = team_encoder.transform(df[["OffTeam"]])
    # df[player_id_cols] = player_encoder.transform(df[player_id_cols])
    return df


def zb_output(df):
    """zb output"""
    type_dict.update({"UsageId": "int32"})
    return df[type_dict.keys()].astype(type_dict)


def zb_pipeline(level="possession"):
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    # team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    # game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    player = pd.read_sql(parse_sql(SQL_PATH["player"], False), engine)
    play = preprocess_play(play)
    collapsed = collapse_plays(play, level=level)
    collapsed = usage_player_id(play, collapsed)
    result = add_dob(collapsed, player)
    result = zb_categorical_encoding(result, from_s3=True)
    return zb_output(result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--level", type=str, default="possession")
    args.add_argument("--output_mode", type=str, default="")
    args = args.parse_args()
    df = zb_pipeline(level=args.level)
    if args.output_mode:
        wr.s3.to_parquet(
            df=df,
            path=S3_FOLDER + f"zb_{args.level}",
            dataset=True,
            mode=args.output_mode,
            # table="proc_play",
            # database="nbastats",
        )
        print("Uploaded parquet file to S3.")
    else:
        print(df.head())
        print(df.dtypes)
    os.system('say "Your Python Program has Finished"')
