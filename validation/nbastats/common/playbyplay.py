from nbastats.common.encoder import encoder_from_s3
import os
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import pandera as pa
from patsy import dmatrix
from pandera.typing import DataFrame
from sklearn.preprocessing import OneHotEncoder
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
        # "off_id": [
        #     "OffPlayer1Id",
        #     "OffPlayer2Id",
        #     "OffPlayer3Id",
        #     "OffPlayer4Id",
        #     "OffPlayer5Id",
        # ],
        # "def_id": [
        #     "DefPlayer1Id",
        #     "DefPlayer2Id",
        #     "DefPlayer3Id",
        #     "DefPlayer4Id",
        #     "DefPlayer5Id",
        # ],
        "off_event": ["P1E", "P2E", "P3E", "P4E", "P5E"],
        "def_event": ["P6E", "P7E", "P8E", "P9E", "P10E"],
        "offdef_event": [
            "P1E",
            "P2E",
            "P3E",
            "P4E",
            "P5E",
            "P6E",
            "P7E",
            "P8E",
            "P9E",
            "P10E",
        ],
        "off_id": ["P1", "P2", "P3", "P4", "P5"],
        "def_id": ["P6", "P7", "P8", "P9", "P10"],
        "offdef_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
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


type_dict = {
    "GameId": "int64",
    "PossCount": "int64",
    "Season": "int64",
    "GameDate": "datetime64",
    "GameType": "int64",
    "OffTeam": "int64",
    "DefTeam": "int64",
    # "Period": "int64",
    "HomeOff": "float32",
    "SecRemainGame": "float32",
    "StartEvent": "str",
    "EndEvent": "str",
    # "HomeScore": "int64",
    # "AwayScore": "int64",
    # "HomePts": "float32",
    # "AwayPts": "float32",
    # "UsageId": "int64",
    "Pts": "float32",
    "Duration": "int64",
    "ShotDistance": "float32",
    "ShotAngle": "float32",
    "ScoreMargin": "float32",
    "P1": "int64",
    "P2": "int64",
    "P3": "int64",
    "P4": "int64",
    "P5": "int64",
    "P6": "int64",
    "P7": "int64",
    "P8": "int64",
    "P9": "int64",
    "P10": "int64",
    "Age1": "float32",
    "Age2": "float32",
    "Age3": "float32",
    "Age4": "float32",
    "Age5": "float32",
    "Age6": "float32",
    "Age7": "float32",
    "Age8": "float32",
    "Age9": "float32",
    "Age10": "float32",
}


@pa.check_input(play_schema)
def preprocess_play(df):
    """preprocess play"""

    # important
    # df = df.sort_values(["GameId", "PlayNum"], ascending=True).reset_index(drop=True)
    df = df.sort_values(["GameDate", "GameId", "PlayNum"], ascending=True).reset_index(
        drop=True
    )

    # home/away to off/def
    # df["OffensiveTeamId"] = np.select(
    #     [df["HomeOff"].isna(), df["HomeOff"] == 1, df["HomeOff"] == 0],
    #     [None, df["HomeTeamId"], df["AwayTeamId"]],
    # )
    # df["DefensiveTeamId"] = np.select(
    #     [df["HomeOff"].isna(), df["HomeOff"] == 1, df["HomeOff"] == 0],
    #     [None, df["AwayTeamId"], df["HomeTeamId"]],
    # )

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
    # df = df.loc[df["HomeOff"].notna()]

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

    return homeaway_to_offdef(df)


# @pa.check_output(agg_schema)
def aggregation_to_game_level(play, game):
    """clean by summation"""
    # play = play.loc[~((play["HomePts"] > 0) & (play["HomeOff"] == 1))]
    # play = play.loc[~((play["AwayPts"] > 0) & (play["HomeOff"] == 0))]
    final_scores = (
        play.groupby("GameId")[["HomeScore", "AwayScore"]]
        .agg(["max", "last"])
        .reset_index(drop=True)
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


def homeaway_to_offdef(df):
    """player homeaway to offdef"""
    df["OffTeam"] = np.where(df["HomeOff"] == 1, df["HomeTeamId"], df["AwayTeamId"])
    df["DefTeam"] = np.where(df["HomeOff"] == 1, df["AwayTeamId"], df["HomeTeamId"])
    df.loc[df["HomeOff"] == 1, column_list("off_id") + column_list("def_id")] = df.loc[
        df["HomeOff"] == 1, column_list("id")
    ].values
    df.loc[df["HomeOff"] == 0, column_list("def_id") + column_list("off_id")] = df.loc[
        df["HomeOff"] == 0, column_list("id")
    ].values
    df.loc[
        df["HomeOff"] == 1, column_list("off_event") + column_list("def_event")
    ] = df.loc[df["HomeOff"] == 1, column_list("event")].values
    df.loc[
        df["HomeOff"] == 0, column_list("def_event") + column_list("off_event")
    ] = df.loc[df["HomeOff"] == 0, column_list("event")].values
    df[column_list("offdef_id")] = df[column_list("offdef_id")].astype("int")
    return df.drop(
        columns=["HomeTeamId", "AwayTeamId"] + column_list("id") + column_list("event")
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


# def players_of_possession(df):
#     """players who use most of the possession"""
#     df["SumSec"] = df.groupby(["GameId", "PossCount"] + column_list("id"))[
#         "SecSinceLastPlay"
#     ].transform(sum)
#     df_id = (
#         df.sort_values(["GameId", "PossCount", "SumSec"], ascending=[True, True, False])
#         .groupby(["GameId", "PossCount"])[column_list("id")]
#         .first()
#     )
#     return pd.merge(
#         df.drop(columns=column_list("id")), df_id, on=["GameId", "PossCount"]
#     )


def encode_player_event(df, from_s3=False):
    """get unique play event"""
    # events = pd.unique(df[column_list("event")].values.ravel())
    # events_series = pd.Series(range(len(events)), events)
    # df[column_list("event")] = df[column_list("event")].apply(
    #     lambda x: events_series[x]
    # )
    df[column_list("offdef_event")] = df[column_list("offdef_event")].replace(
        {None: np.nan}
    )
    if from_s3:
        event_encoder = encoder_from_s3("event")
    else:
        raise NotImplementedError
    for event_col in column_list("offdef_event"):
        df[event_encoder.get_feature_names([event_col])] = event_encoder.transform(
            df[[event_col]]
        ).toarray()
    return df.loc[:, ~df.columns.str.endswith("_nan")]


def collapse_plays(df, level="possession", event=False):
    """collapse play level"""
    # check homeoff consistency

    collapsed_cols = [
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
        "OffTeam",
        "DefTeam",
        "PlayNum",
        "HomeScore",
        "AwayScore",
    ] + column_list("offdef_id")
    agg_dict = {
        "Season": ("Season", "first"),
        "GameDate": ("GameDate", "first"),
        "GameType": ("GameType", "first"),
        "OffTeam": ("OffTeam", "first"),
        "DefTeam": ("DefTeam", "first"),
        # ScoreMargin:("ScoreMargin", "first"),
        "Period": ("Period", "first"),
        "HomeOff": ("HomeOff", "first"),
        "SecRemainGame": ("SecRemainGame", "max"),
        "StartPlayNum": ("PlayNum", "min"),
        "StartEvent": ("StartEvent", "first"),
        "EndEvent": ("EndEvent", "last"),
        "HomeScore": ("HomeScore", "max"),
        "AwayScore": ("AwayScore", "max"),
        "HomePts": ("HomePts", "sum"),
        "AwayPts": ("AwayPts", "sum"),
        "Duration": ("SecSinceLastPlay", "sum"),
        "ShotDistance": ("ShotDistance", "mean"),
        "ShotAngle": ("ShotAngle", "mean"),
    }

    if event:
        event_cols = df.filter(regex="P\d+E_.*").columns.tolist()
        collapsed_cols += event_cols
        agg_dict.update({col: (col, "sum") for col in event_cols})

    collapsed = df.loc[:, collapsed_cols]
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

    collapsed["SecLineup"] = collapsed.groupby(
        ["GameId", "PossCount"] + column_list("offdef_id")
    )["SecSinceLastPlay"].transform("sum")

    result = collapsed.groupby(["GameId", "PossCount"]).agg(**agg_dict)
    players = collapsed.sort_values("SecLineup", ascending=False).drop_duplicates(
        ["GameId", "PossCount"]
    )
    assert len(result.index) == len(players.index)
    result = pd.merge(
        result,
        players[["GameId", "PossCount"] + column_list("offdef_id")],
        on=["GameId", "PossCount"],
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


def build_spline(df):
    """build B spline"""
    splines_config = {
        # "cols": column_list("age"),
        "knots": [25, 33],
        "degree": 3,
    }
    age_col = df[column_list("age")].stack()
    trans = dmatrix(
        "bs(train, knots, degree, include_intercept=False)",
        {"train": age_col}.update(splines_config),
        return_type="dataframe",
    )
    trans = trans.drop(columns="Intercept")
    trans.columns = [f"_{i}" for i in len(trans.columns)]
    return trans


def add_dob(df, player):
    df["Pts"] = np.where(df["HomeOff"] == 1, df["HomePts"], df["AwayPts"])
    # df = df.rename(
    #     columns=dict(
    #         zip(
    #             column_list("off_id") + column_list("def_id"),
    #             column_list("off_id_abbr") + column_list("def_id_abbr"),
    #         )
    #     )
    # )
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


# def common_play(play, level="possession"):
#     """ common processing for play"""
#     df = preprocess_play(play)
#     return df, collapse_plays(df, level=level)


if __name__ == "__main__":
    """main script"""
    print("This is common script.")
