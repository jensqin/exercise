import sys
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy

from bla_python_db_utilities.parser import parse_sql

sys.path.append("./")

from settings import ENGINE_CONFIG, SQL_PATH
from nbastats.common.playbyplay import column_names, common_play


def usage_player_id(df, collapsed):
    """get usage player id"""

    # remove ast as event
    # event_columns = [
    #     f"{x}Player{y}Event" for x in ["Home", "Away"] for y in range(1, 6)
    # ]
    df = df.loc[
        :,
        ["GameId", "PossCount", "HomeOff", "Eventmsgtype"]
        + column_names("id")
        + column_names("event"),
    ]
    df.loc[df["Eventmsgtype"].isin([1, 2, 3, 5]), column_names("event")] = df.loc[
        df["Eventmsgtype"].isin([1, 2, 3, 5]), column_names("event")
    ].replace({"Ast": None})
    df = df.drop(columns="Eventmsgtype")

    # # insert 0 for null playerId
    # id_columns = [f"{x}Player{y}Id" for x in ["Home", "Away"] for y in range(1, 6)]
    # df[column_names("id")] = df[column_names("id")].fillna(0)

    # only one player uses a possession
    assert (
        df.loc[df["HomeOff"] == 0, column_names("away_event")].notna().sum(axis=1) < 2
    ).all()
    assert (
        df.loc[df["HomeOff"] == 1, column_names("home_event")].notna().sum(axis=1) < 2
    ).all()

    UsgId0 = np.where(
        df.loc[df["HomeOff"] == 0, column_names("away_event")].notna(),
        df.loc[df["HomeOff"] == 0, column_names("away_id")],
        0,
    ).sum(axis=1)
    UsgId1 = np.where(
        df.loc[df["HomeOff"] == 1, column_names("home_event")].notna(),
        df.loc[df["HomeOff"] == 1, column_names("home_id")],
        0,
    ).sum(axis=1)

    # df["PlayerId"] = None
    df.loc[df["HomeOff"] == 0, "UsageId"] = UsgId0
    df.loc[df["HomeOff"] == 1, "UsageId"] = UsgId1
    # df["PlayerId"] = df["PlayerId"].fillna(0)
    # df["UsageId"] = np.where(df["HomeOff"] == 1, UsgId1, UsgId0)

    # player_ID_data$poss_game_id <- paste(player_ID_data$GameId, player_ID_data$PossCount)
    # unique_last_possession <- length(player_ID_data$poss_game_id) - match(unique(player_ID_data$poss_game_id), rev(player_ID_data$poss_game_id)) + 1

    # Assign the player_ids of the player responsible for using each possession
    # player_ID_data_reduced <- player_ID_data[unique_last_possession, ]

    # last_play = df.groupby(["GameId", "PossCount"])["UsageId"].last()
    df["UsageId"] = (
        df.loc[df["UsageId"] > 0]
        .groupby(["GameId", "PossCount"])["UsageId"]
        .transform("last")
    )
    df.loc[
        df["HomeOff"] == 1, column_names("off_id") + column_names("def_id")
    ] = df.loc[
        df["HomeOff"] == 1, column_names("home_id") + column_names("away_id")
    ].values
    df.loc[
        df["HomeOff"] == 0, column_names("off_id") + column_names("def_id")
    ] = df.loc[
        df["HomeOff"] == 0, column_names("away_id") + column_names("home_id")
    ].values
    df[column_names("off_id") + column_names("def_id")] = df[
        column_names("off_id") + column_names("def_id")
    ].astype("int")
    # df[column_names("off_id") + column_names("def_id")] = np.where(
    #     df["HomeOff"] == 1,
    #     df[column_names("home_id") + column_names("away_id")].values,
    #     df[column_names("away_id") + column_names("home_id")].values,
    # )
    df = df.loc[
        df[column_names("off_id")]
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
                "HomeOff",
                "Duration",
                "HomePts",
                "AwayPts",
                "ShotDistance",
                "ShotAngle",
            ]
        ],
        df[
            ["GameId", "PossCount"] + column_names("off_id") + column_names("def_id")
        ].drop_duplicates(),
        on=["GameId", "PossCount"],
        how="left",
    )
    return pd.merge(result, poss_usg, on=["GameId", "PossCount"], how="left")

    # map_player_ids_usage <- match(paste(data$GameId, data$PossCount), paste(player_ID_data_reduced$GameId, player_ID_data_reduced$PossCount))
    # data$UsageID <- player_ID_data_reduced[map_player_ids_usage, "PlayerId"]
    # data$UsageID[is.na(data$UsageID)] <- 0

    # one_poss_lineup <- setDT(data_event[
    #   !is.na(data_event$SecSinceLastPlay) &
    #     !is.na(data_event$PossCount) &
    #     data_event$Eventmsgtype != 8,
    #   c("GameId", "PossCount", "HomeLineup", "AwayLineup", "SecSinceLastPlay", "PlayNum")
    # ])[, .(SecSinceLastPlay = sum(SecSinceLastPlay), PlayNum = mean(PlayNum)), by = list(GameId, PossCount, HomeLineup, AwayLineup)]

    # # Add UsageID
    # one_poss_lineup$UsageID <- data$UsageID[match(paste(one_poss_lineup$GameId, one_poss_lineup$PossCount), paste(data$GameId, data$PossCount))]
    # one_poss_lineup$UsageID[is.na(one_poss_lineup$UsageID)] <- 0

    # one_poss_lineup$containsUsg <- 1 * (str_detect(one_poss_lineup$HomeLineup, paste0("^", one_poss_lineup$UsageID, ",")) |
    #   str_detect(one_poss_lineup$HomeLineup, paste0(",", one_poss_lineup$UsageID, "$")) |
    #   str_detect(one_poss_lineup$HomeLineup, paste0(",", one_poss_lineup$UsageID, ",")) |
    #   str_detect(one_poss_lineup$AwayLineup, paste0("^", one_poss_lineup$UsageID, ",")) |
    #   str_detect(one_poss_lineup$AwayLineup, paste0(",", one_poss_lineup$UsageID, "$")) |
    #   str_detect(one_poss_lineup$AwayLineup, paste0(",", one_poss_lineup$UsageID, ","))
    # )
    # one_poss_lineup$containsUsg[one_poss_lineup$UsageID == 0] <- NA

    # check if any poss have bad usg player data
    # mean(one_poss_lineup$containsUsg, na.rm = T) # > 99% of plays the usage player is included in the offensive players on court, other cases are likely data errors

    # one_poss_lineup <- one_poss_lineup[order(one_poss_lineup$GameId, one_poss_lineup$PossCount, -one_poss_lineup$SecSinceLastPlay, one_poss_lineup$PlayNum), ]
    # # Remove rows where Usg player not in lineup
    # one_poss_lineup <- one_poss_lineup[which(one_poss_lineup$containsUsg == 1 |
    # is.na(one_poss_lineup$containsUsg)), ]
    # # Break tie by longest duration on court for the lineup
    # one_poss_lineup <- one_poss_lineup[match(
    # unique(paste(one_poss_lineup$GameId, one_poss_lineup$PossCount)),
    # paste(one_poss_lineup$GameId, one_poss_lineup$PossCount)
    # ), ]


def zb_summarise(df, player):
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
    # Response
    # data$PTS <- ifelse(data$HomeOff == 1, data$HomePts, data$AwayPts)

    # Home court advantage and game situation variables (time remaining and point differential)
    # Add in days rest, and consecutive minutes played varibales ACTION
    # HCA <- data$HomeOff
    # Score_Diff = data$ScoreMargin # perspective of offensive team

    # Time_Remaining <- data$SecRemainGame / 60 / 48

    # birthdates <- player_data[, c("PlayerId", "Dob")]
    # birthdates <- birthdates[order(as.numeric(birthdates$PlayerId)), ]

    # # Offensive Aging Curves
    # age1 <- birthdates[match(players_off[, "P1"], birthdates$PlayerId), "Dob"]
    # age2 <- birthdates[match(players_off[, "P2"], birthdates$PlayerId), "Dob"]
    # age3 <- birthdates[match(players_off[, "P3"], birthdates$PlayerId), "Dob"]
    # age4 <- birthdates[match(players_off[, "P4"], birthdates$PlayerId), "Dob"]
    # age5 <- birthdates[match(players_off[, "P5"], birthdates$PlayerId), "Dob"]
    # players_off <- cbind.data.frame(players_off, age1, age2, age3, age4, age5)

    # # Defensive Aging Curves
    # age6 <- birthdates[match(players_def[, "P6"], birthdates$PlayerId), "Dob"]
    # age7 <- birthdates[match(players_def[, "P7"], birthdates$PlayerId), "Dob"]
    # age8 <- birthdates[match(players_def[, "P8"], birthdates$PlayerId), "Dob"]
    # age9 <- birthdates[match(players_def[, "P9"], birthdates$PlayerId), "Dob"]
    # age10 <- birthdates[match(players_def[, "P10"], birthdates$PlayerId), "Dob"]
    # players_def <- cbind.data.frame(players_def, age6, age7, age8, age9, age10)

    # data$GameDate <- game_data$GameDate[match(data$GameId, game_data$GameId)]

    # # Convert Birthdates to ages
    # data$GameDate <- as.Date(data$GameDate, format = "%Y-%m-%d")
    # players_off[, "age1"] <- as.integer(data$GameDate - players_off[, "age1"]) / 365.25
    # players_off[, "age2"] <- as.integer(data$GameDate - players_off[, "age2"]) / 365.25
    # players_off[, "age3"] <- as.integer(data$GameDate - players_off[, "age3"]) / 365.25
    # players_off[, "age4"] <- as.integer(data$GameDate - players_off[, "age4"]) / 365.25
    # players_off[, "age5"] <- as.integer(data$GameDate - players_off[, "age5"]) / 365.25
    # players_def[, "age6"] <- as.integer(data$GameDate - players_def[, "age6"]) / 365.25
    # players_def[, "age7"] <- as.integer(data$GameDate - players_def[, "age7"]) / 365.25
    # players_def[, "age8"] <- as.integer(data$GameDate - players_def[, "age8"]) / 365.25
    # players_def[, "age9"] <- as.integer(data$GameDate - players_def[, "age9"]) / 365.25
    # players_def[, "age10"] <- as.integer(data$GameDate - players_def[, "age10"]) / 365.25

    # PTS_scored <- data$PTS

    # # do chance level here instead of possession and this wont be an issue (in theory)
    # # cap rare values
    # # PTS_scored[PTS_scored > 3] <- 3

    # # offensive and defensive team seasons
    # data$HomeTeamId <- game_data$HomeTeamId[match(floor(data$GameId), game_data$GameId)]
    # data$AwayTeamId <- game_data$AwayTeamId[match(floor(data$GameId), game_data$GameId)]
    # data$OffensiveTeamId <- ifelse(data$HomeOff == 1, data$HomeTeamId, data$AwayTeamId)
    # data$DefensiveTeamId <- ifelse(data$HomeOff == 0, data$HomeTeamId, data$AwayTeamId)

    # # playoffs indicator
    # data$postseason <- as.integer(substr(data$GameId, 1, 1))
    # postseason <- ifelse(data$postseason == 4, 1, 0)

    # Control for capped possession duration
    # chance level should reduce the skew
    # duration <- data$SecSinceLastPlay
    # # duration[duration < 5] <- 5
    # # duration[duration > 25] <- 25

    # proc_data <- cbind.data.frame(
    # # y_exp = NA, # need to make separate pipline to write expected points to the database
    # y = data$PTS, HomeAway = data$HomeOff, ScoreDiff = Score_Diff, Duration = duration, Season = data$Season, Playoffs = postseason, Time_Remaining,
    # OffTeam = data$OffensiveTeamId, DefTeam = data$DefensiveTeamId,
    # players_off[, 1:10],
    # players_def[, 1:10]
    # )

    # proc_data$UsageID <- data$UsageID
    # proc_data$PossCount <- data$PossCount
    # proc_data$GameId <- data$GameId

    # proc_data <- proc_data[rowSums(is.na(proc_data)) == 0, ]

    # playerids <- sort(as.integer(as.character(unique(unlist(proc_data[, c("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")])))))
    # playerids_map <- 0:(length(playerids) - 1)

    # teamids <- sort(as.integer(as.character(unique(unlist(proc_data[, c("OffTeam", "DefTeam")])))))
    # teamids_map <- 0:(length(teamids) - 1)

    # assign new ids to players
    # for (i in 1:10) {
    #   proc_data[, paste0("P", i)] <- playerids_map[match(proc_data[, paste0("P", i)], as.character(playerids))]
    # }

    # # new ids to teams
    # proc_data[, "OffTeam"] <- teamids_map[match(proc_data[, "OffTeam"], teamids)]
    # proc_data[, "DefTeam"] <- teamids_map[match(proc_data[, "DefTeam"], teamids)]

    # team_map <- data.frame(teamids_map, teamids, name = teams$Name[match(teamids, teams$TeamId)])
    # player_map <- data.frame(playerids_map, playerids, name = player_data$Name[match(playerids, player_data$PlayerId)])

    # groups <- c(
    #   0, 0, 0, 0, 0, 0, 1, 2,
    #   rep(3, 5),
    #   rep(4, 5),
    #   rep(5, 5),
    #   rep(6, 5),
    #   7
    # )

    # groups <- data.frame(colname = colnames(proc_data)[-c(1, dim(proc_data)[2] - 1, dim(proc_data)[2])], group = groups)


def zb_pipeline():
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    # team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    # game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    player = pd.read_sql(parse_sql(SQL_PATH["player"], False), engine)
    play, possession = common_play(play)
    possession = usage_player_id(play, possession)
    return zb_summarise(possession, player)


if __name__ == "__main__":
    zb_pipeline()
