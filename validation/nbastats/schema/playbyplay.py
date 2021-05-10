import pandas as pd
from pandas import Timestamp
import pandera as pa
from pandera.typing import DataFrame, Series, DateTime, String


class InputPlaySchema(pa.SchemaModel):
    """input playbyplay data schema"""

    GameId: Series[int] = pa.Field(ge=20000000)
    GameDate: Series[DateTime] = pa.Field(
        ge=Timestamp("2000-01-01 00:00:00"), coerce=True
    )
    GameType: Series[int] = pa.Field(isin=[0, 1])
    PlayNum: Series[int] = pa.Field(ge=0)
    Eventmsgtype: Series[int] = pa.Field(ge=0)
    Season: Series[int] = pa.Field(ge=2000)
    SecRemainPeriod: Series[int] = pa.Field(ge=0)
    ScoreMargin: Series[int]
    HomeOff: Series[int] = pa.Field(isin=[0, 1], nullable=True)
    StartEvent: Series[String]
    EndEvent: Series[String]
    PossCount: Series[int] = pa.Field(ge=0, nullable=True)
    SecSinceLastPlay: Series[int] = pa.Field(ge=0)
    Eventmsgactiontype: Series[int] = pa.Field(ge=0)
    ShotDistance: Series[float] = pa.Field(ge=0, nullable=True)
    ShotAngle: Series[float] = pa.Field(ge=0, le=180, nullable=True)

    # regex columns
    TeamId: Series[int] = pa.Field(alias=".*TeamId", regex=True)
    PlayerId: Series[int] = pa.Field(alias=".*Player\dId", regex=True, nullable=True)
    PlayerEvent: Series[String] = pa.Field(
        alias=".*Player\dEvent", regex=True, nullable=True
    )
    NumPlayers: Series[int] = pa.Field(alias=".*NumPlayers", regex=True, ge=0)
    Fouls: Series[int] = pa.Field(alias=".*Fouls", regex=True, ge=0)
    Score: Series[int] = pa.Field(alias=".*Score", regex=True, ge=0)
    Pos: Series[float] = pa.Field(alias="Pos.", regex=True, nullable=True)


class PreProcessedPlaySchema(InputPlaySchema):
    """preprocessed play schema"""

    PossCount: Series[int] = pa.Field(ge=0)
    OffensiveTeamId: Series[int] = pa.Field(ge=0)
    DefensiveTeamId: Series[int] = pa.Field(ge=0)
    HomePts: Series[int] = pa.Field(ge=0)
    AwayPts: Series[int] = pa.Field(ge=0)


# class CollapsedPlaySchema(pa.SchemaModel):
#     """collapsed play schema"""
#     pass
