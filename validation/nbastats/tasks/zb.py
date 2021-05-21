import sys
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
from nbastats.common.playbyplay import column_list, common_play


def usage_player_id(df, collapsed):
    """get usage player id"""

    # remove ast as event
    # event_columns = [
    #     f"{x}Player{y}Event" for x in ["Home", "Away"] for y in range(1, 6)
    # ]
    df = df.loc[
        :,
        ["GameId", "PossCount", "HomeOff", "Eventmsgtype"]
        + column_list("id")
        + column_list("event"),
    ]
    df.loc[df["Eventmsgtype"].isin([1, 2, 3, 5]), column_list("event")] = df.loc[
        df["Eventmsgtype"].isin([1, 2, 3, 5]), column_list("event")
    ].replace({"Ast": None})
    df = df.drop(columns="Eventmsgtype")

    # # insert 0 for null playerId
    # id_columns = [f"{x}Player{y}Id" for x in ["Home", "Away"] for y in range(1, 6)]
    # df[column_names("id")] = df[column_names("id")].fillna(0)

    # only one player uses a possession
    assert (
        df.loc[df["HomeOff"] == 0, column_list("away_event")].notna().sum(axis=1) < 2
    ).all()
    assert (
        df.loc[df["HomeOff"] == 1, column_list("home_event")].notna().sum(axis=1) < 2
    ).all()

    UsgId0 = np.where(
        df.loc[df["HomeOff"] == 0, column_list("away_event")].notna(),
        df.loc[df["HomeOff"] == 0, column_list("away_id")],
        0,
    ).sum(axis=1)
    UsgId1 = np.where(
        df.loc[df["HomeOff"] == 1, column_list("home_event")].notna(),
        df.loc[df["HomeOff"] == 1, column_list("home_id")],
        0,
    ).sum(axis=1)

    # df["PlayerId"] = None
    df.loc[df["HomeOff"] == 0, "UsageId"] = UsgId0
    df.loc[df["HomeOff"] == 1, "UsageId"] = UsgId1
    # df["PlayerId"] = df["PlayerId"].fillna(0)
    # df["UsageId"] = np.where(df["HomeOff"] == 1, UsgId1, UsgId0)

    df["UsageId"] = (
        df.loc[df["UsageId"] > 0]
        .groupby(["GameId", "PossCount"])["UsageId"]
        .transform("last")
    )
    df.loc[df["HomeOff"] == 1, column_list("off_id") + column_list("def_id")] = df.loc[
        df["HomeOff"] == 1, column_list("home_id") + column_list("away_id")
    ].values
    df.loc[df["HomeOff"] == 0, column_list("off_id") + column_list("def_id")] = df.loc[
        df["HomeOff"] == 0, column_list("away_id") + column_list("home_id")
    ].values
    df[column_list("off_id") + column_list("def_id")] = df[
        column_list("off_id") + column_list("def_id")
    ].astype("int")
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

    result = pd.merge(
        collapsed[
            [
                "GameId",
                "PossCount",
                "GameDate",
                "GameType",
                "OffTeam",
                "DefTeam",
                "HomeOff",
                "ScoreMargin",
                "Duration",
                "HomePts",
                "AwayPts",
                "ShotDistance",
                "ShotAngle",
            ]
        ],
        df[
            ["GameId", "PossCount"] + column_list("off_id") + column_list("def_id")
        ].drop_duplicates(),
        on=["GameId", "PossCount"],
        how="left",
    )
    return pd.merge(result, poss_usg, on=["GameId", "PossCount"], how="left")


def zb_summarise(df, player):
    df["Pts"] = np.where(df["HomeOff"] == 1, df["HomePts"], df["AwayPts"])
    df = df.rename(
        columns=dict(
            zip(
                column_list("off_id") + column_list("def_id"),
                column_list("off_id_abbr") + column_list("def_id_abbr"),
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

    df[column_list("age")] = df[column_list("age")].apply(
        lambda s: (df["GameDate"] - s).dt.days / 365.25
    )
    return df


def categorical_encoding(df, from_s3=False):
    """encode categorical variables"""
    player_id_cols = column_list("off_id_abbr") + column_list("def_id_abbr")
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


def zb_pipeline(level="possession"):
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    # team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    # game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    player = pd.read_sql(parse_sql(SQL_PATH["player"], False), engine)
    play, collapsed = common_play(play, level=level)
    collapsed = usage_player_id(play, collapsed)
    result = zb_summarise(collapsed, player)
    return categorical_encoding(result, from_s3=True)


if __name__ == "__main__":
    df = zb_pipeline(level="chance")
    wr.s3.to_parquet(
        df=df,
        path=S3_FOLDER + "zb_play_example",
        dataset=True,
        mode="overwrite",
        # table="proc_play",
        # database="nbastats",
    )
    print("Uploaded parquet file to S3.")
