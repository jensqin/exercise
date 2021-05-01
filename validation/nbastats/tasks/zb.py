import os
import numpy as np
import pandas as pd
import pandera as pa
import sqlalchemy

from bla_python_db_utilities.parser import parse_sql
from settings import ENGINE_CONFIG, SQL_PATH
from nbastats.common.playbyplay import column_names, convert_homeaway_to_offdef


def usage_player_id(df):
    """get usage player id"""

    # remove ast as event
    # event_columns = [
    #     f"{x}Player{y}Event" for x in ["Home", "Away"] for y in range(1, 6)
    # ]
    df.loc[df["Eventmsgtype"].isin([1, 2, 3, 5]), column_names("event"),] = df.loc[
        df["Eventmsgtype"].isin([1, 2, 3, 5]), column_names("event"),
    ].replace({"Ast", None})

    # # insert 0 for null playerId
    # id_columns = [f"{x}Player{y}Id" for x in ["Home", "Away"] for y in range(1, 6)]
    df[column_names("id")] = df[column_names("id")].fillna(0)
    player_id = df

    UsgId0 = (
        player_id.loc[player_id["HomeOff"] == 0, column_names("away_event")].notnan()
        * player_id.loc[player_id["HomeOff"] == 0, column_names("away_id")]
    ).sum()
    UsgId1 = (
        player_id.loc[player_id["HomeOff"] == 1, column_names("home_event")].notnan()
        * player_id.loc[player_id["HomeOff"] == 1, column_names("home_id")]
    ).sum()

    player_id["PlayerId"] = None
    player_id.loc[player_id["HomeOff"] == 0, "PlayerId"] = UsgId0
    player_id.loc[player_id["HomeOff"] == 1, "PlayerId"] = UsgId1
    player_id["PlayerId"] = player_id["PlayerId"].fillna(0)

    return player_id

    # player_ID_data$poss_game_id <- paste(player_ID_data$GameId, player_ID_data$PossCount)
    # unique_last_possession <- length(player_ID_data$poss_game_id) - match(unique(player_ID_data$poss_game_id), rev(player_ID_data$poss_game_id)) + 1

    # Assign the player_ids of the player responsible for using each possession
    # player_ID_data_reduced <- player_ID_data[unique_last_possession, ]
    # player_ID_data_reduced <- player_ID_data_reduced[paste(player_ID_data_reduced$GameId, player_ID_data_reduced$PossCount) %in%
    # paste(data$GameId, data$PossCount), ]
    # map_player_ids_usage <- match(paste(data$GameId, data$PossCount), paste(player_ID_data_reduced$GameId, player_ID_data_reduced$PossCount))
    # data$UsageID <- player_ID_data_reduced[map_player_ids_usage, "PlayerId"]
    # data$UsageID[is.na(data$UsageID)] <- 0

    # one_poss_lineup <- setDT(data_event[
    #   !is.na(data_event$SecSinceLastPlay) &
    #     !is.na(data_event$PossCount) &
    #     data_event$Eventmsgtype != 8,
    #   c("GameId", "PossCount", "HomeLineup", "AwayLineup", "SecSinceLastPlay", "PlayNum")
    # ])[, .(SecSinceLastPlay = sum(SecSinceLastPlay), PlayNum = mean(PlayNum)), by = list(GameId, PossCount, HomeLineup, AwayLineup)]
    # setDF(one_poss_lineup)

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


def encode_play_event(df):
    """get unique play event"""
    events = pd.unique(df[column_names("event")].values.ravel())
    events_series = pd.Series(range(len(events)), events)
    df[column_names("event")] = df[column_names("event")].apply(
        lambda x: events_series[x]
    )
    return df

def summarize_data_frame(data):
    pass
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

def processing():
    """data processing"""

    engine = sqlalchemy.create_engine(ENGINE_CONFIG["DEV_NBA.url"])
    team = pd.read_sql(parse_sql(SQL_PATH["team"], False), engine)
    game = pd.read_sql(parse_sql(SQL_PATH["game"], False), engine)
    play = pd.read_sql(parse_sql(SQL_PATH["play"], False), engine)
    return convert_homeaway_to_offdef(play)


if __name__ == "__main__":
    processing()
